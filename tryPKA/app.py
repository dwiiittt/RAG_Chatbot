"""
Flask Movie Chatbot — main app (Groq API)
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify

from rag_engine import init_engine, retrieve_movies, expand_query
from chat_engine import chat

app = Flask(__name__)
app.secret_key = "movie-chatbot-secret-2024"

_sessions: dict = {}

# ── Init RAG on startup ─────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "imdb_movies.csv")
init_engine(CSV_PATH)
print("✅ Engine ready.")

# ── Groq API Key — ganti di sini atau set: export GROQ_API_KEY=gsk_... ─────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_msg = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400
    if not GROQ_API_KEY or GROQ_API_KEY == "ISI_GROQ_API_KEY_KAMU_DI_SINI":
        return jsonify({"error": "Groq API key belum diisi di app.py"}), 500

    history = _sessions.get(session_id, [])
    expanded = expand_query(user_msg, history)
    movies = retrieve_movies(expanded, top_k=7)

    try:
        answer = chat(user_msg, history, movies, GROQ_API_KEY)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": answer})
    _sessions[session_id] = history

    return jsonify({
        "answer": answer,
        "session_id": session_id,
        "movies_retrieved": [m["title"] for m in movies],
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    session_id = request.json.get("session_id")
    if session_id and session_id in _sessions:
        _sessions[session_id] = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)