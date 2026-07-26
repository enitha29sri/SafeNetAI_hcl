# 🛡 SafeNet AI — Digital Safety Assistant

A production-quality Generative AI application that helps everyday users stay
safe online through an AI chat assistant, a Retrieval-Augmented Generation
(RAG) cybersecurity knowledge base, and a suite of practical safety tools.

Final Year Engineering Project.

---

## ✨ Features

- 💬 **AI Chat** — Conversational assistant powered by Google Gemini, grounded by RAG when a knowledge base is indexed
- 📚 **RAG Knowledge Base** — Grounded answers from curated cybersecurity PDFs (PyMuPDF + FAISS + `all-MiniLM-L6-v2`)
- 🚨 **Scam Message Analyzer** — Local heuristic engine + optional AI explanation for pasted messages
- 🌐 **Website Safety Advisor** — Offline, structural URL risk analysis (no live requests to the target site)
- 🎓 **Cybersecurity Learning Mode** — Five structured lessons covering fundamentals
- 💡 **Daily Cyber Tip** — Deterministic tip-of-the-day, persisted per calendar date
- 📝 **Interactive Quiz** — 5-question scored quiz with per-session history
- 📊 **Dashboard** — Plotly analytics: chat volume, quiz trends, feedback sentiment
- 🗂 **Chat History & Export** — Messages persist to SQLite; full transcript downloadable as Markdown
- 👍 **Feedback** — Thumbs up/down on assistant replies, persisted for the Dashboard
- 🌓 **Markdown Rendering, Typing Animation, Suggested Prompts** — throughout the Chat page

> **Not included in this build:** Password Strength Analyzer (Phase 5) and Privacy Permission Advisor
> (Phase 8) were intentionally skipped at the project owner's request. Their sidebar entries show a
> clean "coming soon" state and can be added later following the same pattern as the Scam Analyzer
> or Website Safety Advisor (heuristic checker in `utils/`, orchestration service in `services/`,
> a render function in `app.py`).

---

## 🏗 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python 3.11 |
| LLM | Google Gemini API (via LangChain) |
| Orchestration | LangChain |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | FAISS |
| Relational DB | SQLite |
| PDF Parsing | PyMuPDF |
| Charts | Plotly |
| Config | python-dotenv |

---

## 📁 Project Structure

```
SafeNetAI/
├── app.py                     # Streamlit entry point: nav, theme, all page routing
├── config.py                  # Centralized settings (env-driven)
├── chatbot.py                 # Gemini chat orchestration
├── database.py                # SQLite access layer (chat, quiz, feedback, settings, tips)
├── rag.py                     # RAG pipeline (load -> chunk -> embed -> retrieve)
├── prompts.py                 # Chat prompt templates
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── utils/
│   ├── risk_analyzer.py       # Shared RiskLevel/RiskFlag/RiskResult primitives
│   ├── scam_detector.py       # Heuristic scam/phishing pattern detector
│   ├── website_checker.py     # Heuristic offline URL safety checker
│   ├── helpers.py             # UI helpers: theme loader, typing animation, copy button
│   └── logger.py              # Centralized loguru configuration
├── models/
│   ├── embedding_model.py     # SentenceTransformer wrapped as a LangChain Embeddings
│   ├── llm.py                 # GeminiLLM wrapper (LangChain ChatGoogleGenerativeAI)
│   └── memory.py              # Bounded conversation memory
├── services/
│   ├── scam_service.py        # Scam Analyzer orchestration (heuristics + optional AI)
│   ├── website_service.py     # Website Advisor orchestration (heuristics + optional AI)
│   ├── learning_service.py    # Static Learning Mode lesson content
│   ├── tip_service.py         # Daily Cyber Tip rotation + persistence
│   └── quiz_service.py        # Quiz question bank, scoring, persistence
├── database/
│   └── sqlite.db              # created at runtime
├── data/
│   ├── pdfs/                  # source knowledge-base PDFs (add your own here)
│   └── vector_store/          # persisted FAISS index (built automatically)
└── assets/
    └── css/
        └── style.css          # dark theme design tokens
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.11
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### 2. Clone & create a virtual environment

```bash
git clone <your-repo-url>
cd SafeNetAI
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` and set your real `GEMINI_API_KEY`.

### 5. Verify configuration

```bash
python config.py
```

You should see:

```
SafeNet AI v1.0.0 configuration OK.
Base directory: /path/to/SafeNetAI
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

On first launch:
- `database/sqlite.db` is created automatically with all required tables.
- If you drop PDFs into `data/pdfs/`, the first Chat page load will build a FAISS index
  under `data/vector_store/` (this requires internet access to download the
  `all-MiniLM-L6-v2` model the first time). Without any PDFs, Chat still works —
  it simply skips knowledge-base grounding.

---

## 🧪 Testing

Every module in this project was verified as it was built:

```bash
# Configuration
python config.py

# Database layer (creates and exercises all 5 tables)
python -c "from database import Database; db = Database(); print('Database OK')"

# Heuristic detectors (no API key needed)
python -c "
from utils.scam_detector import ScamDetector
from utils.website_checker import WebsiteChecker
print(ScamDetector().analyze('Dear customer, verify your account now: http://bit.ly/x').level)
print(WebsiteChecker().analyze('http://paypal-secure-login.tk/verify').level)
"

# Full app (manual click-through)
streamlit run app.py
```

For automated UI regression testing, this project uses Streamlit's built-in
`AppTest` (headless script execution — catches real runtime exceptions,
not just syntax errors):

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py")
at.run()
assert not at.exception
```

---

## 🚀 Deployment

### Streamlit Community Cloud (simplest)
1. Push this repository to GitHub (`.env` stays out of it — it's git-ignored).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set the main
   file to `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_real_key_here"
   ```
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

### Self-hosted (VM / Docker)
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY
streamlit run app.py --server.port 8501 --server.headless true
```
Put a reverse proxy (nginx/Caddy) in front for TLS if exposing publicly.

**Note on persistence:** `database/sqlite.db` and `data/vector_store/` are local files.
On platforms with ephemeral filesystems (e.g. some container hosts), mount a persistent
volume at the project root, or point `DATABASE_PATH` / `VECTOR_STORE_DIR` in `.env` at a
mounted volume.

---

## ⚠️ Known Limitations

- **No user accounts.** Each browser tab gets an anonymous `session_id` generated at
  runtime. Chat history and quiz history are scoped to that session and are lost on a
  full page reload (a new session starts). The Dashboard's aggregate stats (total
  messages, average quiz %, feedback sentiment) span *all* sessions and persist across
  reloads, since those are read from SQLite directly.
- **Password Strength Analyzer and Privacy Permission Advisor are not implemented** —
  skipped by project scope. Their nav entries show a "coming soon" state.
- **RAG requires internet access** the first time it runs, to download the
  `all-MiniLM-L6-v2` embedding model from Hugging Face. After that first download, it's
  cached locally and works offline.
- **Website Safety Advisor is structural-only** — it analyzes the URL string itself
  (protocol, domain, path patterns) and never fetches the target page, so it cannot
  detect malicious page *content*, only common structural red flags.

---

## 🔒 Security Notes

- Passwords are never a feature of this build (the Password Checker was skipped), so no
  password handling code exists to audit.
- Scam messages and URLs submitted to the analyzers are **never logged** — only
  exception messages are logged, never the analyzed content itself.
- All secrets are read from `.env`, which is git-ignored by default.
- User input is validated before being passed to any model, heuristic, or database call.

---

## 🗺 Build Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup, config, docs | ✅ Done |
| 2 | Gemini integration, prompts, memory | ✅ Done |
| 3 | Full RAG pipeline | ✅ Done |
| 4 | Streamlit UI shell & theme | ✅ Done |
| 5 | Password Strength Analyzer | ⏭️ Skipped |
| 6 | Scam Message Analyzer | ✅ Done |
| 7 | Website Safety Advisor | ✅ Done |
| 8 | Privacy Permission Advisor | ⏭️ Skipped |
| 9 | Learning Module, Daily Tips, Quiz | ✅ Done |
| 10 | Dashboard & Analytics | ✅ Done |
| 11 | Chat History, Export, Feedback | ✅ Done |
| 12 | Testing, bug fixes, optimization, deployment | ✅ Done |

---

## 📄 License

Educational / Final Year Project use.
