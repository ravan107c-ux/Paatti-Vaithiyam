"""
Paatti Vaithiyam — backend API
--------------------------------
A small Flask service that stores oral home-remedy knowledge as a
symptom -> herb -> preparation -> safety knowledge graph, and serves
it to the frontend.

Run:
    pip install -r requirements.txt
    python app.py
Then open frontend/index.html in a browser (it calls this API at
http://127.0.0.1:5050).
"""

import os
import hmac
import re
import sqlite3
from datetime import datetime, timezone
from flask import Flask, g, jsonify, request, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash
from healthcare_match import analyze_remedy

DB_PATH = os.environ.get("PAATTI_DB_PATH", os.path.join(os.path.dirname(__file__), "paatti.db"))

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or os.urandom(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower() == "true",
)

# ---------------------------------------------------------------------------
# CORS (handled by hand so we don't depend on flask-cors being installed)
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Review-Token"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    phone         TEXT NOT NULL UNIQUE,
    email         TEXT NOT NULL UNIQUE,
    location      TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS remedies (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    title_ta      TEXT,
    symptom       TEXT NOT NULL,
    herbs         TEXT NOT NULL,          -- comma separated
    preparation   TEXT NOT NULL,
    who_for       TEXT,                   -- e.g. "children", "adults", "pregnant - avoid"
    dosage        TEXT,
    elder_name    TEXT NOT NULL,
    village       TEXT NOT NULL,
    category      TEXT NOT NULL,
    verified      INTEGER DEFAULT 0,      -- community-verified count
    safety_flag   TEXT,                   -- NULL, 'caution', or 'review'
    safety_note   TEXT,
    created_at    TEXT NOT NULL,
    submitted_by  INTEGER REFERENCES users(id),
    publication_status TEXT NOT NULL DEFAULT 'approved'
);

CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    remedy_id   INTEGER NOT NULL REFERENCES remedies(id) ON DELETE CASCADE,
    author      TEXT,
    rating      INTEGER NOT NULL,
    comment     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shops (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    town    TEXT NOT NULL,
    address TEXT,
    phone   TEXT,
    hours   TEXT
);

CREATE TABLE IF NOT EXISTS app_metadata (
    name  TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# A tiny "AI extraction" + safety layer
# ---------------------------------------------------------------------------
# In the real product, steps 2-5 from the brief (speech-to-text, structure
# extraction, graph building, pharmacology cross-check) would call actual
# models / a curated interaction database. Here we simulate that pipeline
# with rule-based keyword matching so the whole flow is runnable offline,
# end to end, without external API keys.

KNOWN_HERBS = [
    "neem", "veppilai", "tulsi", "துளசி", "hibiscus", "செம்பருத்தி",
    "pepper", "milagu", "turmeric", "manjal", "ginger", "inji",
    "honey", "coconut oil", "betel leaf", "vetiver", "vilvam",
    "fenugreek", "vendhayam", "garlic", "poondu", "castor oil",
    "amla", "nellikai", "curry leaves", "karuveppilai",
]

# caution rules: herb keyword -> (who it's dangerous/unsuitable for, note)
SAFETY_RULES = {
    "pepper": ("infants under 1 year", "High doses of pepper can irritate an infant's airway and stomach lining."),
    "milagu": ("infants under 1 year", "High doses of pepper can irritate an infant's airway and stomach lining."),
    "honey": ("infants under 1 year", "Honey carries a botulism risk for babies under 12 months and should never be given to them."),
    "castor oil": ("pregnant women", "Castor oil can stimulate uterine contractions and is generally avoided in pregnancy."),
    "garlic": ("people on blood-thinning medicine", "Garlic in medicinal amounts can add to the effect of blood-thinning medication."),
    "poondu": ("people on blood-thinning medicine", "Garlic in medicinal amounts can add to the effect of blood-thinning medication."),
    "vetiver": ("no major interaction on file", None),
}

SYMPTOM_KEYWORDS = [
    "cough", "cold", "fever", "hair fall", "dandruff", "indigestion",
    "acidity", "wound", "skin", "joint pain", "headache", "sleep",
    "constipation", "sore throat", "toothache", "burns",
]


def extract_structure(raw_text: str):
    """Very small rule-based stand-in for the 'speech text -> structured
    fields' AI step described in the brief. Looks for known herb and
    symptom keywords inside free text."""
    text = raw_text.lower()
    found_herbs = [h for h in KNOWN_HERBS if h in text]
    found_symptom = next((s for s in SYMPTOM_KEYWORDS if s in text), None)
    return found_herbs, found_symptom


def safety_check(herbs_text: str, who_for: str):
    """Cross-checks each mentioned herb against SAFETY_RULES. Returns
    (flag, note) where flag is None / 'caution' / 'review'."""
    text = (herbs_text or "").lower()
    notes = []
    for herb, (risk_group, note) in SAFETY_RULES.items():
        if herb in text and note:
            notes.append(f"{herb.title()}: caution for {risk_group}. {note}")
    if notes:
        return "caution", " | ".join(notes)
    return None, None


# ---------------------------------------------------------------------------
# Seed data — reflects the example remedies from the project brief
# ---------------------------------------------------------------------------
def seed_if_empty(db):
    count = db.execute("SELECT COUNT(*) c FROM remedies").fetchone()["c"]
    if count > 0:
        return
    now = datetime.now(timezone.utc).isoformat()
    remedies = [
        (
            "A neem leaf rinse", "வேப்பிலை தண்ணீர்", "skin",
            "neem, veppilai", "Fresh neem leaves washed and steeped in warm water; cooled fully before use.",
            "adults and children (external use only)", "Use as a cooled rinse, once daily",
            "Lakshmi Paatti", "Madurai", "Skin & body", 14, None, None, now,
        ),
        (
            "Hibiscus hair oil", "செம்பருத்தி எண்ணெய்", "hair fall",
            "hibiscus, செம்பருத்தி, coconut oil", "Hibiscus flowers and leaves slow-infused in warmed coconut oil, cooled, strained.",
            "adults and children", "Massage into scalp 2-3 times a week",
            "Meenakshi Paatti", "Melur", "Hair & scalp", 9, None, None, now,
        ),
        (
            "Tulsi evening brew", "துளசி காய்ச்சல் கஷாயம்", "cough",
            "tulsi, துளசி, pepper, honey", "Tulsi leaves boiled with a pinch of pepper, strained, honey stirred in once warm (not hot).",
            "children over 1 year and adults", "One small cup, once in the evening",
            "Meenakshi Paatti", "Melur village", "Seasonal comfort", 15, "caution",
            "Pepper: caution for infants under 1 year. High doses of pepper can irritate an infant's airway and stomach lining. | Honey: caution for infants under 1 year. Honey carries a botulism risk for babies under 12 months and should never be given to them.",
            now,
        ),
        (
            "Ginger-turmeric throat soother", "இஞ்சி மஞ்சள் கஷாயம்", "sore throat",
            "ginger, inji, turmeric, manjal, honey", "Grated ginger and a pinch of turmeric boiled in water, strained, honey added once warm.",
            "adults and children over 1 year", "Sip warm, twice a day",
            "Rajammal Paatti", "Thanjavur", "Seasonal comfort", 11, "caution",
            "Honey: caution for infants under 1 year. Honey carries a botulism risk for babies under 12 months and should never be given to them.",
            now,
        ),
        (
            "Fenugreek water for indigestion", "வெந்தய தண்ணீர்", "indigestion",
            "fenugreek, vendhayam", "A teaspoon of fenugreek seeds soaked overnight in water; the water is drunk on an empty stomach.",
            "adults", "One small cup, morning only",
            "Kamakshi Paatti", "Kumbakonam", "Digestion", 7, None, None, now,
        ),
        (
            "Betel leaf and castor poultice", "வெற்றிலை ஆமணக்கு", "joint pain",
            "betel leaf, castor oil", "Betel leaves lightly warmed with castor oil and pressed onto the joint.",
            "adults (avoid during pregnancy)", "Apply once at night",
            "Valliammal Paatti", "Sivagangai", "Body care", 5, "caution",
            "Castor Oil: caution for pregnant women. Castor oil can stimulate uterine contractions and is generally avoided in pregnancy.",
            now,
        ),
        (
            "Amla and curry leaf hair rinse", "நெல்லிக்காய் கறிவேப்பிலை", "dandruff",
            "amla, nellikai, curry leaves, karuveppilai", "Amla and curry leaves boiled together, the cooled water strained and used as a final hair rinse.",
            "adults and children", "Use after regular wash, 2 times a week",
            "Lakshmi Paatti", "Madurai", "Hair & scalp", 8, None, None, now,
        ),
    ]
    db.executemany(
        """INSERT INTO remedies
           (title, title_ta, symptom, herbs, preparation, who_for, dosage,
            elder_name, village, category, verified, safety_flag, safety_note, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        remedies,
    )

    reviews = [
        (1, "Priya", 5, "Used this for my daughter's heat rash, worked gently.", now),
        (1, "Arun", 5, "Exactly how my own paatti used to make it.", now),
        (2, "Divya", 5, "Scalp feels so much calmer after a few weeks.", now),
        (3, "Karthik", 4, "Good for a mild cough, my son liked the taste too.", now),
        (5, "Meena", 4, "Helped with bloating after heavy meals.", now),
    ]
    db.executemany(
        "INSERT INTO reviews (remedy_id, author, rating, comment, created_at) VALUES (?,?,?,?,?)",
        reviews,
    )

    shops = [
        ("Sri Selvavinayagar Nattu Marunthu Kadai", "Madurai", "West Masi Street, Madurai", "+91 98430 11122", "9:00 AM - 8:30 PM"),
        ("Amman Herbals", "Thanjavur", "Big Bazaar Street, Thanjavur", "+91 97510 44556", "8:30 AM - 9:00 PM"),
        ("Kaveri Nattu Marunthu Shop", "Kumbakonam", "TSR Big Street, Kumbakonam", "+91 96290 77812", "9:00 AM - 8:00 PM"),
        ("Muthu Herbal Store", "Melur", "Main Bazaar, Melur", "+91 90031 55672", "9:00 AM - 7:30 PM"),
    ]
    db.executemany(
        "INSERT INTO shops (name, town, address, phone, hours) VALUES (?,?,?,?,?)",
        shops,
    )
    db.commit()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    remedy_columns = {row["name"] for row in db.execute("PRAGMA table_info(remedies)")}
    if "submitted_by" not in remedy_columns:
        db.execute("ALTER TABLE remedies ADD COLUMN submitted_by INTEGER REFERENCES users(id)")
    if "publication_status" not in remedy_columns:
        db.execute("ALTER TABLE remedies ADD COLUMN publication_status TEXT NOT NULL DEFAULT 'approved'")
    seed_if_empty(db)
    migrated = db.execute(
        "SELECT 1 FROM app_metadata WHERE name='legacy_remedies_need_review'"
    ).fetchone()
    if migrated is None:
        db.execute(
            "UPDATE remedies SET publication_status='pending' "
            "WHERE id > 7 AND submitted_by IS NULL"
        )
        db.execute(
            "INSERT INTO app_metadata (name, value) VALUES (?, ?)",
            ("legacy_remedies_need_review", datetime.now(timezone.utc).isoformat()),
        )
    db.commit()
    db.close()


def json_object():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def text_value(data, key, default=""):
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip()


init_db()


def public_user(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "phone": row["phone"],
        "email": row["email"],
        "location": row["location"],
    }


def current_user():
    user_id = session.get("user_id")
    if user_id is None:
        return None
    user = get_db().execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if user is None:
        session.clear()
    return user


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def remedy_to_dict(row, db):
    reviews = db.execute(
        "SELECT author, rating, comment, created_at FROM reviews WHERE remedy_id=? ORDER BY id DESC",
        (row["id"],),
    ).fetchall()
    avg = None
    if reviews:
        avg = round(sum(r["rating"] for r in reviews) / len(reviews), 1)
    return {
        "id": row["id"],
        "title": row["title"],
        "title_ta": row["title_ta"],
        "symptom": row["symptom"],
        "herbs": [h.strip() for h in row["herbs"].split(",")],
        "preparation": row["preparation"],
        "who_for": row["who_for"],
        "dosage": row["dosage"],
        "elder_name": row["elder_name"],
        "village": row["village"],
        "category": row["category"],
        "verified": row["verified"],
        "safety_flag": row["safety_flag"],
        "safety_note": row["safety_note"],
        "publication_status": row["publication_status"],
        "rating_avg": avg,
        "rating_count": len(reviews),
        "reviews": [dict(r) for r in reviews],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "paatti-vaithiyam-api"})


@app.route("/")
def frontend():
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required"}), 400
    try:
        name = text_value(data, "name")
        phone = text_value(data, "phone")
        email = text_value(data, "email").lower()
        location = text_value(data, "location")
        password = text_value(data, "password")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    if not all((name, phone, email, location, password)):
        return jsonify({"error": "Name, phone, email, location, and password are required"}), 400
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return jsonify({"error": "Enter a valid email address"}), 400
    if not re.fullmatch(r"\+?[0-9][0-9\s().-]{5,19}", phone):
        return jsonify({"error": "Enter a valid phone number"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    db = get_db()
    try:
        cursor = db.execute(
            "INSERT INTO users (name, phone, email, location, password_hash, created_at) VALUES (?,?,?,?,?,?)",
            (name, phone, email, location, generate_password_hash(password), datetime.now(timezone.utc).isoformat()),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account already uses that email or phone number"}), 409

    session.clear()
    session["user_id"] = cursor.lastrowid
    user = db.execute("SELECT * FROM users WHERE id=?", (cursor.lastrowid,)).fetchone()
    return jsonify({"user": public_user(user)}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required"}), 400
    try:
        email = text_value(data, "email").lower()
        password = text_value(data, "password")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    user = get_db().execute("SELECT * FROM users WHERE lower(email)=?", (email,)).fetchone()
    if user is None or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Email or password is incorrect"}), 401
    session.clear()
    session["user_id"] = user["id"]
    return jsonify({"user": public_user(user)})


@app.route("/api/auth/me")
def auth_me():
    user = current_user()
    return jsonify({"authenticated": user is not None, "user": public_user(user) if user else None})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"authenticated": False})


@app.route("/api/review/remedies")
def pending_remedies():
    review_token = os.environ.get("REMEDY_REVIEW_TOKEN")
    supplied_token = request.headers.get("X-Review-Token", "")
    if not review_token:
        return jsonify({"error": "Review workflow is not configured"}), 503
    if not hmac.compare_digest(supplied_token, review_token):
        return jsonify({"error": "Review authorization required"}), 403
    db = get_db()
    rows = db.execute(
        "SELECT * FROM remedies WHERE publication_status='pending' ORDER BY id"
    ).fetchall()
    pending = [remedy_to_dict(row, db) for row in rows]
    dataset_path = os.environ.get("HEALTHCARE_DATASET_PATH")
    for remedy in pending:
        remedy["healthcare_analysis"] = analyze_remedy(remedy, dataset_path)
    return jsonify(pending)


@app.route("/api/remedies")
def list_remedies():
    db = get_db()
    q = request.args.get("q", "").strip().lower()
    category = request.args.get("category", "").strip().lower()

    rows = db.execute(
        "SELECT * FROM remedies WHERE publication_status='approved' ORDER BY verified DESC, id ASC"
    ).fetchall()
    results = []
    for row in rows:
        haystack = " ".join([
            row["title"], row["title_ta"] or "", row["symptom"],
            row["herbs"], row["category"], row["elder_name"], row["village"],
        ]).lower()
        if q and not any(term in haystack for term in q.split()):
            continue
        if category and category not in row["category"].lower():
            continue
        results.append(remedy_to_dict(row, db))
    return jsonify(results)


@app.route("/api/remedies/<int:remedy_id>")
def get_remedy(remedy_id):
    db = get_db()
    row = db.execute("SELECT * FROM remedies WHERE id=?", (remedy_id,)).fetchone()
    user = current_user()
    if row is None or (row["publication_status"] != "approved" and row["submitted_by"] != (user["id"] if user else None)):
        return jsonify({"error": "Remedy not found"}), 404
    return jsonify(remedy_to_dict(row, db))


@app.route("/api/remedies", methods=["POST"])
def add_remedy():
    """Accepts either already-structured fields, or a single 'raw_text'
    field (simulating a transcribed voice note) that we run through the
    extraction + safety pipeline ourselves."""
    data = json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required"}), 400
    user = current_user()
    if user is None:
        return jsonify({"error": "Sign in to share a remedy"}), 401
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()

    try:
        raw_text = text_value(data, "raw_text")
        herbs = text_value(data, "herbs")
        symptom = text_value(data, "symptom")
        title = text_value(data, "title")
        title_ta = text_value(data, "title_ta")
        preparation = text_value(data, "preparation")
        who_for = text_value(data, "who_for")
        dosage = text_value(data, "dosage")
        elder_name = text_value(data, "elder_name", user["name"]) or user["name"]
        village = text_value(data, "village", user["location"]) or user["location"]
        category = text_value(data, "category", "Community submitted") or "Community submitted"
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    if raw_text and not (herbs and symptom):
        found_herbs, found_symptom = extract_structure(raw_text)
        herbs = herbs or ", ".join(found_herbs) or "not detected — please add manually"
        symptom = symptom or found_symptom or "general"

    title = title or f"{symptom.title()} remedy from {elder_name}"
    flag, note = safety_check(herbs, who_for)

    cur = db.execute(
        """INSERT INTO remedies
           (title, title_ta, symptom, herbs, preparation, who_for, dosage,
            elder_name, village, category, verified, safety_flag, safety_note,
            created_at, submitted_by, publication_status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            title,
            title_ta or None,
            symptom or "general",
            herbs or "not specified",
            preparation or raw_text or "Shared verbally; preparation notes pending.",
            who_for or None,
            dosage or None,
            elder_name,
            village,
            category,
            0,
            flag,
            note,
            now,
            user["id"],
            "pending",
        ),
    )
    db.commit()
    row = db.execute("SELECT * FROM remedies WHERE id=?", (cur.lastrowid,)).fetchone()
    return jsonify(remedy_to_dict(row, db)), 201


@app.route("/api/review/remedies/<int:remedy_id>", methods=["POST"])
def review_remedy(remedy_id):
    review_token = os.environ.get("REMEDY_REVIEW_TOKEN")
    supplied_token = request.headers.get("X-Review-Token", "")
    if not review_token:
        return jsonify({"error": "Review workflow is not configured"}), 503
    if not hmac.compare_digest(supplied_token, review_token):
        return jsonify({"error": "Review authorization required"}), 403
    data = json_object()
    status = data.get("status") if data else None
    if not isinstance(status, str) or status not in {"approved", "rejected"}:
        return jsonify({"error": "status must be approved or rejected"}), 400
    db = get_db()
    result = db.execute(
        "UPDATE remedies SET publication_status=? WHERE id=? AND publication_status='pending'",
        (status, remedy_id),
    )
    if result.rowcount == 0:
        return jsonify({"error": "Pending remedy not found"}), 404
    db.commit()
    return jsonify({"id": remedy_id, "publication_status": status})


@app.route("/api/remedies/<int:remedy_id>/verify", methods=["POST"])
def verify_remedy(remedy_id):
    """Lets another elder / practitioner add a community confirmation."""
    db = get_db()
    db.execute("UPDATE remedies SET verified = verified + 1 WHERE id=? AND publication_status='approved'", (remedy_id,))
    db.commit()
    row = db.execute(
        "SELECT * FROM remedies WHERE id=? AND publication_status='approved'", (remedy_id,)
    ).fetchone()
    if row is None:
        return jsonify({"error": "Remedy not found"}), 404
    return jsonify(remedy_to_dict(row, db))


@app.route("/api/remedies/<int:remedy_id>/reviews", methods=["POST"])
def add_review(remedy_id):
    user = current_user()
    if user is None:
        return jsonify({"error": "Sign in to leave a comment"}), 401
    data = json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required"}), 400
    rating_value = data.get("rating", 0)
    if isinstance(rating_value, bool) or not isinstance(rating_value, (int, str)):
        return jsonify({"error": "rating must be an integer between 1 and 5"}), 400
    try:
        rating = int(rating_value)
    except ValueError:
        return jsonify({"error": "rating must be an integer between 1 and 5"}), 400
    if rating < 1 or rating > 5:
        return jsonify({"error": "rating must be between 1 and 5"}), 400
    try:
        comment = text_value(data, "comment")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    db = get_db()
    row = db.execute(
        "SELECT id FROM remedies WHERE id=? AND publication_status='approved'", (remedy_id,)
    ).fetchone()
    if row is None:
        return jsonify({"error": "Remedy not found"}), 404
    db.execute(
        "INSERT INTO reviews (remedy_id, author, rating, comment, created_at) VALUES (?,?,?,?,?)",
        (remedy_id, user["name"], rating, comment, datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    full = db.execute("SELECT * FROM remedies WHERE id=?", (remedy_id,)).fetchone()
    return jsonify(remedy_to_dict(full, db)), 201


@app.route("/api/graph")
def graph():
    """Returns a symptom -> herb knowledge-graph as nodes + edges so the
    frontend can render the 'searchable knowledge graph' from the brief."""
    db = get_db()
    rows = db.execute(
        "SELECT id, symptom, herbs, title, safety_flag FROM remedies WHERE publication_status='approved'"
    ).fetchall()

    nodes = {}
    edges = []
    for row in rows:
        s_id = f"symptom::{row['symptom'].lower()}"
        if s_id not in nodes:
            nodes[s_id] = {"id": s_id, "label": row["symptom"].title(), "type": "symptom"}
        for herb in [h.strip() for h in row["herbs"].split(",")]:
            if not herb or herb == "not specified":
                continue
            h_id = f"herb::{herb.lower()}"
            if h_id not in nodes:
                nodes[h_id] = {"id": h_id, "label": herb.title(), "type": "herb"}
            edges.append({
                "source": s_id,
                "target": h_id,
                "remedy_id": row["id"],
                "remedy_title": row["title"],
                "caution": bool(row["safety_flag"]),
            })
    return jsonify({"nodes": list(nodes.values()), "edges": edges})


@app.route("/api/symptoms")
def symptoms():
    db = get_db()
    rows = db.execute(
        "SELECT DISTINCT symptom FROM remedies WHERE publication_status='approved' ORDER BY symptom"
    ).fetchall()
    return jsonify([r["symptom"] for r in rows])


@app.route("/api/shops")
def shops():
    db = get_db()
    town = request.args.get("town", "").strip().lower()
    rows = db.execute("SELECT * FROM shops ORDER BY name").fetchall()
    results = [dict(r) for r in rows]
    if town:
        filtered = [r for r in results if town in r["town"].lower()]
        results = filtered or results  # fall back to showing all as a prototype behaviour
    return jsonify(results)


if __name__ == "__main__":
    print("Paatti Vaithiyam API running at http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=os.environ.get("FLASK_DEBUG") == "1")
