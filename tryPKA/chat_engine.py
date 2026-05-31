"""
Chat logic: Groq API (gratis) dengan model LLaMA 3.3 70B.
Strict grounding prompt — tidak halusinasi.
"""

import re
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Model Groq gratis tersedia:
# - "llama-3.3-70b-versatile"  ← paling pintar (recommended)
# - "llama-3.1-8b-instant"     ← paling cepat
# - "gemma2-9b-it"             ← alternatif
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a movie information assistant. You ONLY answer questions about movies using information from the [MOVIE CONTEXT] block provided.

STRICT RULES — follow these without exception:
1. ONLY use facts from [MOVIE CONTEXT]. Never invent or assume facts.
2. For crew/cast questions: copy the exact crew field from context. Do not guess or add names not in the data.
3. For follow-up questions like "who are the cast?" or "what's the budget?" — refer to the EXACT movie(s) mentioned in [MOVIE CONTEXT] and in the conversation history.
4. If user says "from that list" or "from those movies" or similar — you MUST only pick from movies already mentioned in your PREVIOUS assistant message. Extract those titles, look them up in [MOVIE CONTEXT], compare their scores, and pick the highest one.
5. NEVER recommend or mention a movie not present in [MOVIE CONTEXT] for this turn.
6. If information is not in [MOVIE CONTEXT] or conversation history, say: "Informasi ini tidak tersedia di database saya."
7. Answer in Bahasa Indonesia unless the user writes in another language.
8. Be concise, friendly, and structured. Use emojis sparingly.
9. For "best from the list" requests: explicitly state which movie has the highest score and explain why based on the data.
10. When listing movies, always include: title, genre, score, and a brief description.

FORMAT for movie lists:
🎬 **[Title]** (Score: XX/100)
- Genre: ...
- Synopsis: ... (1-2 sentences max)

FORMAT for detailed single movie:
🎬 **[Title]**
- Rilis: ...
- Genre: ...
- Skor: XX/100
- Bahasa: ... | Negara: ...
- Budget: ... | Pendapatan: ...
- Pemeran & Kru: [EXACT crew field from context]
- Sinopsis: ..."""


def build_context_block(movies: list) -> str:
    if not movies:
        return "[MOVIE CONTEXT: No relevant movies found in database]"
    parts = []
    for i, m in enumerate(movies):
        parts.append(
            f"[Movie {i+1}]\n"
            f"Title: {m['title']}\n"
            f"Original Title: {m['orig_title']}\n"
            f"Release: {m['release']}\n"
            f"Genre: {m['genre']}\n"
            f"Score: {m['score']}/100\n"
            f"Status: {m['status']}\n"
            f"Language: {m['language']} | Country: {m['country']}\n"
            f"Budget: {m['budget']} | Revenue: {m['revenue']}\n"
            f"Crew (exact): {m['crew']}\n"
            f"Overview: {m['overview']}"
        )
    return "[MOVIE CONTEXT]\n" + "\n\n".join(parts) + "\n[END MOVIE CONTEXT]"


def extract_mentioned_titles(text: str) -> list:
    return re.findall(r'\*\*([^*]+)\*\*', text)


def chat(query: str, history: list, movies: list, api_key: str) -> str:
    # Detect "from that list" follow-up
    followup_phrases = [
        "dari list", "dari daftar", "dari yang tadi", "dari itu", "dari situ",
        "mana yang paling", "which is the best", "yang terbaik dari",
        "yang paling bagus", "most recommended", "highest rated dari",
        "dari rekomendasi", "dari hasil", "pilih yang terbaik",
    ]
    is_list_followup = any(p in query.lower() for p in followup_phrases)

    if is_list_followup and history:
        last_bot = [m['content'] for m in history if m['role'] == 'assistant']
        if last_bot:
            prev_titles = extract_mentioned_titles(last_bot[-1])
            if prev_titles:
                from rag_engine import get_movies_by_titles
                anchored = get_movies_by_titles(prev_titles)
                if anchored:
                    movies = anchored

    context_block = build_context_block(movies)
    user_msg_content = f"{context_block}\n\nPertanyaan pengguna: {query}"

    trimmed = history[-12:]  # max 6 turns
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + trimmed
        + [{"role": "user", "content": user_msg_content}]
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]