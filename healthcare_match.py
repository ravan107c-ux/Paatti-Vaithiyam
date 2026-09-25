"""Informational text matching against a locally downloaded medicine CSV."""

import csv
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RELEVANT_COLUMNS = (
    "medicine type",
    "generic",
    "active ingredient",
    "therapeutic class",
    "indication",
    "pharmacology",
    "interaction",
    "side effect",
    "pregnancy",
    "precaution",
    "warning",
    "dosage",
)
DISPLAY_COLUMNS = ("generic title", "generic", "brand name", "medicine type", "indications")


def _normalise_header(value):
    return " ".join(str(value or "").casefold().replace("_", " ").replace("-", " ").split())


def _record_text(record):
    parts = []
    for key, value in record.items():
        header = _normalise_header(key)
        if value and any(column in header for column in RELEVANT_COLUMNS):
            parts.append(str(value).strip())
    return " ".join(parts)


@lru_cache(maxsize=3)
def _load_index(path, modified_at):
    del modified_at
    records = []
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for record in csv.DictReader(source):
            text = _record_text(record)
            if text:
                records.append((record, text))
    if not records:
        return None
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=75000, stop_words="english")
    matrix = vectorizer.fit_transform(text for _, text in records)
    return records, vectorizer, matrix


def analyze_remedy(remedy, dataset_path):
    """Return candidate catalog matches; this is not a clinical safety verdict."""
    if not dataset_path:
        return {"status": "dataset_not_configured", "matches": []}

    path = Path(dataset_path).expanduser()
    if not path.is_file():
        return {"status": "dataset_file_missing", "matches": []}

    try:
        index = _load_index(str(path.resolve()), path.stat().st_mtime_ns)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        return {"status": "dataset_unreadable", "message": str(error), "matches": []}
    if index is None:
        return {"status": "dataset_empty", "matches": []}

    records, vectorizer, matrix = index
    query_parts = []
    for key in ("title", "symptom", "herbs", "preparation", "who_for"):
        value = remedy.get(key)
        if isinstance(value, (list, tuple)):
            query_parts.extend(str(item) for item in value)
        elif value:
            query_parts.append(str(value))
    query = " ".join(query_parts).strip()
    if not query:
        return {"status": "no_remedy_text", "matches": []}

    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    ranked = scores.argsort()[::-1]
    matches = []
    for index in ranked:
        score = float(scores[index])
        if score <= 0 or len(matches) == 3:
            break
        record = records[index][0]
        display = {}
        for key, value in record.items():
            header = _normalise_header(key)
            if any(column in header for column in DISPLAY_COLUMNS) and value:
                display[key] = str(value)[:500]
        matches.append({"similarity": round(score, 3), "record": display})

    return {
        "status": "candidate_matches" if matches else "no_candidate_match",
        "model": "TF-IDF cosine similarity",
        "matches": matches,
        "note": "Text similarity is not evidence of efficacy or safety; clinician review is required.",
    }