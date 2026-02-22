# 🧠 Multimodal AI Assistant

Upload **PDFs, Word docs, Excel sheets, Images, Audio, and Video** — then ask anything using **Groq** (free ⚡).

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key
```bash
copy .env.example .env
# Open .env and paste your key: GROQ_API_KEY=gsk_...
# Get free key at: https://console.groq.com
```

### 3. Run (2 terminals)

**Terminal 1 — Backend:**
```bash
set GROQ_API_KEY=gsk_your_key_here
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501** in your browser.

## 📁 Supported Files

| Type | Formats |
|------|---------|
| Documents | PDF, DOCX, XLSX |
| Images | PNG, JPG, WEBP |
| Audio | MP3, WAV, M4A, OGG |
| Video | MP4, AVI, MOV, MKV |

## 🤖 Models (all free on Groq)

- **llama-4-scout-17b** — vision (images, video frames)
- **llama-3.3-70b** — text documents
- **whisper-large-v3-turbo** — audio/video transcription

## 🏗️ Structure

```
multimodal-ai-assistant/
├── backend/main.py       # FastAPI endpoints
├── frontend/app.py       # Streamlit UI
├── utils/parsers.py      # File parsers
├── utils/context.py      # Context builder
├── requirements.txt
└── .env.example
```

## 👤 Author
**Ravi Teja Kesagani** — [LinkedIn](https://linkedin.com/in/raviteja5)
