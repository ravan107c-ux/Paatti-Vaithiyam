# Paatti Vaithiyam — வழிவழி வைத்தியம்

A living home-remedy notebook: elders' spoken knowledge, turned into a
searchable symptom → herb → safety knowledge graph. New submissions stay
private until an authorized reviewer approves them.

This is a small two-part app:

- **`app.py`** — a Python (Flask) REST API backed by SQLite. It stores
  remedies, sends recorded audio to Hugging Face Whisper for transcription,
  optionally translates non-English transcripts with Gemini, and serves the
  symptom/herb knowledge graph as JSON.
- **`index.html`** — a single themed HTML/CSS/JS page that calls that API
  directly with `fetch` (no build step needed).

## Run it

**1. Install dependencies and start the backend**

```bash
pip install -r requirements.txt
python app.py
```

This starts the API at `http://127.0.0.1:5050` and creates `paatti.db`
(SQLite) with seed data the first time the app starts — the same example
remedies from the brief (tulsi cough brew, hibiscus hair oil, neem
rinse, etc.) plus eight additional informational demo entries. Existing
databases are topped up with any missing demo entries on startup.

**2. Open the app**

With the backend running, open `http://127.0.0.1:5050`. Flask serves the
frontend from the same origin so account sessions work correctly.

## Multilingual Voice Search and Translation (Gemini AI)

Voice recording is powered directly by Google Gemini API:
- **Hero Voice Search**: Tap the 🎙️ microphone button in the search bar. Speak in **any language** (Tamil, Hindi, Telugu, Malayalam, Bengali, English, etc.). Gemini AI transcribes and translates your speech into English in real-time, inputs it into the search box, and searches remedies automatically.
- **Share a Remedy**: In the remedy sharing modal, elders and family members can speak naturally in any language. Gemini transcribes and translates the remedy into English for easy review and archival.
- **API Key Setup**: Add `GEMINI_API_KEY=your_key` to `.env`. You can get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Naatu Marundhu Kadai (நாட்டு மருந்து கடை) & Google Maps

Search traditional herbal shops by city across Tamil Nadu:
- **City Search**: Type any city name (e.g. Madurai, Chennai, Coimbatore, Trichy, Salem, Tirunelveli, Thanjavur, Kumbakonam, Erode, Vellore...) or click one of the quick city chips.
- **Pre-Loaded Local Shops**: Displays authentic local shops, verified addresses, operating hours, and clickable contact numbers (`tel:` links for instant calling).
- **Google Maps Redirection**: Click **"🗺️ View on Google Maps"** to open live Google Maps centered on your city showing all nearby Naattu Marundhu Kadai shops. Each shop card also includes direct links for Google Maps view and turn-by-turn directions.

## Accounts and Review

Create an account with name, phone, email, location, and a password of
at least 8 characters. Passwords are stored as hashes. Sign in with email
and password. Remedy submissions require an authenticated session.

Before saving a submission, the server asks Gemini with Google Search grounding
whether a named ingredient has a general or traditional-use connection to the
stated problem. Submissions found unrelated or without supporting sources are
not saved; if Gemini is unavailable, users must retry after it is configured.
Passing this screen is not evidence of effectiveness or safety. Accepted
submissions remain private and still require an authorized human reviewer.
The protected review queue includes Gemini's concise rationale and source links.

New submissions are private with `publication_status: "pending"`. They
remain out of public search, symptoms, and the knowledge graph until an
authorized reviewer approves them. Set strong `FLASK_SECRET_KEY` and
`REMEDY_REVIEW_TOKEN` environment values before starting the app. The
review API uses `X-Review-Token` to list pending remedies at
`GET /api/review/remedies` and approve or reject an item with
`POST /api/review/remedies/<id>` and `{"status":"approved"}` or
`{"status":"rejected"}`.
For an HTTPS deployment, also set `SESSION_COOKIE_SECURE=true`. Flask
debug mode is disabled unless `FLASK_DEBUG=1` is set.

## Kaggle Matching

The optional reviewer analysis uses TF-IDF and cosine similarity against a
local medicine catalog CSV. The Kaggle candidate
[Global Medicine Directory & Healthcare Dataset](https://www.kaggle.com/datasets/utsh0dey/global-medicine-directory-and-healthcare-dataset)
lists an MIT license, describes 26,205 medicine products (including 913
herbal/nutraceutical products), and attributes its catalog to MedEx in
Bangladesh. It is a medicine catalog, not patient records or clinical
trial evidence; review the source and license terms before deployment.

Create a Kaggle API token in your Kaggle account and save its JSON file at
`%USERPROFILE%\.kaggle\kaggle.json`. Do not commit that file or share its
contents. Then download the CSV and configure the app from Windows
Command Prompt:

```bat
pip install kaggle
kaggle datasets download -d utsh0dey/global-medicine-directory-and-healthcare-dataset -p data --unzip
set HEALTHCARE_DATASET_PATH=data\medicine_dataset.csv
set FLASK_SECRET_KEY=replace-with-a-long-random-secret
set REMEDY_REVIEW_TOKEN=replace-with-a-separate-random-token
python app.py
```

The protected `GET /api/review/remedies` queue returns up to three text
matches and similarity scores for each pending item. Similarity is not a
probability or clinical endorsement. A reviewer must approve or reject
each remedy explicitly; there is no automatic publication based on a
model match. Without a downloaded CSV, review still works and the analysis
reports that the dataset is unavailable.

## What's actually implemented

- **Search & browse** — remedies are fetched live from `/api/remedies`,
  filterable by free text and category.
- **Knowledge graph** — `/api/graph` returns symptom and herb nodes with
  edges between them; the frontend draws this as an SVG web (herb ↔
  symptom connections), with cautioned edges drawn in a different color.
- **Safety notes** — `SAFETY_RULES` in `app.py` is an illustrative
  rule-based table. It is not a clinical cross-check and does not verify
  efficacy or safety.
- **Voice-first submission** — "Share a remedy" records audio with the
  browser's `MediaRecorder` API, transcribes it with Hugging Face Whisper,
  and uses Gemini for English translation when the selected spoken language
  is Tamil, Hindi, or Malayalam. The transcript remains editable before
  submission. The submitted text is run through `extract_structure()`, a
  small keyword-based stand-in for the "AI reads the text and pulls out
  symptom + herbs" step.
- **Community verification** — "Confirm this remedy" increments a
  `verified` counter per remedy, standing in for the brief's "other
  elders can confirm or add nuance" idea.
- **Reviews** — star rating + comment per remedy, stored and averaged
  server-side.
- **Nattu Marunthu Kadai finder** — a small seeded shop directory,
  filterable by town.

## Where this is a prototype, not a product

- The safety table is illustrative, not a licensed pharmacology
  database — do not treat it as medical advice, and don't ship it as
  one without a real clinical review process.
- Speech-to-text and translation require the optional Hugging Face and
  Gemini API integrations described above. `extract_structure()` remains a
  rule-based prototype, and `SAFETY_RULES` is not a maintained herb–drug
  interaction database; both require clinical and product review before
  deployment.
- Kaggle data downloading requires the user's Kaggle credentials and is
  configured manually. Clinical validation remains a separate
  human-review responsibility.

## Theme

The frontend deliberately avoids a generic dashboard look: deep granite
stone, temple bronze and madder-red, gold leaf accents, and an
olai-chuvadi (palm-leaf manuscript) background texture, with headings
set in Cormorant Garamond and Tamil text in Noto Serif Tamil.
