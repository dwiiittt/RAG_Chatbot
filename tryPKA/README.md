# 🎬 CineBot — Movie RAG Chatbot (Flask)

Chatbot film berbasis RAG yang **tidak halusinasi** — jawaban 100% dari database IMDB ~10K film.

## Fitur Utama

- ✅ **Anti-halusinasi** — hanya jawab dari data yang ada
- ✅ **Context-aware follow-up** — "dari list itu mana yang terbaik?" benar-benar mengambil dari list sebelumnya
- ✅ **Crew/cast akurat** — langsung dari field data, tidak mengarang
- ✅ **Multi-turn memory** — percakapan nyambung per sesi
- ✅ **Flask + UI aesthetic** — dark cinematic theme

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Siapkan dataset

Letakkan `imdb_movies.csv` di folder yang sama dengan `app.py`.

### 3. Set API Key

Opsi A — environment variable:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Opsi B — isi langsung di sidebar UI saat buka browser.

### 4. Jalankan

```bash
python app.py
```

Buka http://localhost:5000

## Cara Kerja

```
User query
  → expand_query() (enriched with history keywords)
  → FAISS retrieval (top-7 movies)
  → Detect "from that list" intent
    → if yes: re-anchor to titles from last bot message
  → Claude Sonnet + strict grounding prompt
  → Structured answer
```

## File Structure

```
movie_chatbot/
├── app.py              # Flask routes
├── rag_engine.py       # FAISS index, retrieval, query expansion
├── chat_engine.py      # Claude API call, context building
├── templates/
│   └── index.html      # Cinematic dark UI
├── requirements.txt
├── imdb_movies.csv     # Dataset
└── README.md
```

## Perbaikan dari versi Gradio sebelumnya

| Masalah lama | Solusi baru |
|---|---|
| "dari list itu" → jawab film lain | Detect intent → re-anchor ke titles di pesan bot sebelumnya |
| Crew/cast ngawur | Inject raw `crew` field ke context, sistem prompt mewajibkan copy exact |
| LLM halusinasi | Strict system prompt: ONLY use [MOVIE CONTEXT], no invention |
| UI polosan | Flask + dark cinematic theme |
