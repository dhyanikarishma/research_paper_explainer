# 📄 Research Paper Explainer AI

An AI research assistant that ingests an academic PDF and turns it into an
**ELI5 + technical summary, flashcards, a quiz, a research-gap analysis, and
a grounded Q&A chat** — powered by **Large Language Models** and **vector
search (Retrieval-Augmented Generation)**.

> Upload a paper → understand it in minutes → export your notes.

---

## ✨ Features

| Feature | What it does |
|---|---|
| 📝 **Summary** | "Explain Like I'm 15" + section-by-section summary + key contribution |
| 🃏 **Flashcards** | Auto-generated study cards (question → answer) |
| ❓ **Quiz** | Interactive multiple-choice quiz with scoring + explanations |
| 🔬 **Research Gaps** | Peer-reviewer-style limitations, open questions, future work |
| 💬 **Chat (RAG)** | Ask anything; answers are grounded in the paper via vector search |
| 📥 **Export** | Download all generated notes as a single Markdown file |

---

## 🧠 How it works (architecture)

```
PDF ──▶ Text extraction ──▶ Chunking ──▶ Embeddings ──▶ Vector store
 (PyMuPDF)              (overlapping)   (Gemini)      (NumPy cosine search)
                                                            │
User question ──▶ Embed question ──▶ Similarity search ─────┘
                                          │
                                  Top-k relevant chunks
                                          │
                                          ▼
                                  Gemini LLM ──▶ Grounded answer
```

**Retrieval-Augmented Generation (RAG)** means the model answers using the
paper's *actual text* (retrieved by vector similarity) instead of relying on
its memory — which keeps answers accurate and reduces hallucination.

---

## 🛠️ Tech stack

- **Python 3.11.9**
- **Streamlit** — web UI (pure Python)
- **Google Gemini** — text generation + embeddings
- **PyMuPDF** — PDF text extraction
- **NumPy** — vector similarity search (the RAG core)
- **python-dotenv** — local secret management

---

## 🚀 Run it locally

```bash
# 1. Clone the repo
git clone https://github.com/dhyanikarishma/research-paper-explainer.git
cd research-paper-explainer

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
cp .env.example .env
# then open .env and paste your key after GEMINI_API_KEY=

# 5. Launch
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

### Get a free Gemini API key
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with a Google account → **Create API key**
3. Copy it into your `.env` file.

---

## ☁️ Deploy free on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **New app** → pick your repo.
3. Set **Main file path** to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_real_key_here"
   ```
5. **Deploy.** You'll get a public URL to share on your resume/LinkedIn.

---

## 🔧 Configuration

All tunables live in `src/config.py` (or override via `.env`):
`GENERATION_MODEL` (default `gemini-2.5-flash`), `EMBEDDING_MODEL`
(default `models/gemini-embedding-001`), `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.

> **Note:** Google occasionally renames models. If you hit a 404 "model not
> found" error, run the "List models" snippet below to see what your key
> supports, then update the names in `.env`.

### List models your key can use
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_KEY")
for m in genai.list_models():
    print(m.name)
```

---

## 📁 Project structure

```
research-paper-explainer/
├── app.py                # Streamlit UI
├── requirements.txt
├── .env.example
├── .streamlit/config.toml
└── src/
    ├── config.py         # settings + API key loading
    ├── pdf_utils.py      # PDF → text
    ├── chunking.py       # text → overlapping chunks
    ├── llm.py            # Gemini wrapper (generate + embed)
    ├── vector_store.py   # embeddings + cosine similarity search
    └── features.py       # summary, flashcards, quiz, gaps, Q&A, export
```

---

## 📝 License

MIT — built by Karishma Dhyani.
