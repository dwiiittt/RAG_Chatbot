# ============================================================
# app.py — Movie RAG Chatbot (Flask)
# Converted from movie_rag_chatbot_v3_fixed.ipynb
# Compatible with index.html (/api/chat & /api/reset)
# ============================================================

import os
import json
import re
import uuid
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

# ============================================================
# KONFIGURASI
# ============================================================

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

ENV_PATH = os.path.join(PROJECT_PATH, 'env')
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    print(f"[OK] .env dimuat dari: {ENV_PATH}")
else:
    print(f"[!] .env tidak ditemukan di {ENV_PATH}")

# ── API Keys ──────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv('GROQ_API_KEY',  '')
TMDB_API_KEY   = os.getenv('TMDB_API_KEY',  '')
OMDB_API_KEY   = os.getenv('OMDB_API_KEY',  '')

# ── Model Config ──────────────────────────────────────────────
LLM_MODEL          = 'llama-3.3-70b-versatile'
EMBEDDING_MODEL    = 'BAAI/bge-small-en-v1.5'
CHROMA_PERSIST_DIR = os.path.join(PROJECT_PATH, 'chroma_movie_db')
MOVIE_CSV_PATH     = os.path.join(PROJECT_PATH, 'imdb_movies.csv')

# ── Search Config ─────────────────────────────────────────────
DEFAULT_RESULT_LIMIT    = 5
MAX_SEMANTIC_CANDIDATES = 20
MEMORY_WINDOW_TURNS     = 10

print('✅ Konfigurasi selesai')
print(f'   LLM         : {LLM_MODEL}')
print(f'   Embedding   : {EMBEDDING_MODEL}')
print(f'   ChromaDB    : {CHROMA_PERSIST_DIR}')
print(f'   TMDB Key    : {"✅ Set" if TMDB_API_KEY else "❌ Belum diset"}')
print(f'   OMDb Key    : {"✅ Set" if OMDB_API_KEY else "❌ Belum diset"}')

# ============================================================
# CELL 3 — DATA LOADING & VECTOR STORE
# ============================================================

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class MovieDataLoader:
    COLUMN_ALIASES = {
        'title':    ['names', 'title', 'movie_title', 'name', 'film_title', 'Series_Title'],
        'year':     ['date_x', 'year', 'release_year', 'release_date', 'Released_Year'],
        'genre':    ['genre', 'genres', 'Genre'],
        'rating':   ['score', 'rating', 'imdb_rating', 'vote_average', 'IMDB_Rating'],
        'overview': ['overview', 'description', 'plot', 'synopsis', 'Overview'],
        'crew':     ['crew', 'director', 'directors', 'Director'],
        'cast':     ['cast', 'actors', 'stars', 'Star1', 'Actors'],
        'orig_title': ['orig_title', 'original_title'],
        'country':  ['country', 'production_countries'],
        'language': ['orig_lang', 'original_language', 'language'],
        'status':   ['status', 'Status'],
        'budget':   ['budget_x', 'budget'],
        'revenue':  ['revenue', 'Revenue'],
    }

    def load(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
        return self._normalize(df)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip() for c in df.columns]
        col_lower = {c.lower(): c for c in df.columns}

        rename_map = {}
        for target, candidates in self.COLUMN_ALIASES.items():
            for cand in candidates:
                if cand in df.columns:
                    rename_map[cand] = target
                    break
                elif cand.lower() in col_lower:
                    rename_map[col_lower[cand.lower()]] = target
                    break

        df = df.rename(columns=rename_map)

        required = ['title', 'year', 'genre', 'rating', 'overview', 'crew']
        optional = ['cast', 'orig_title', 'country', 'language', 'status', 'budget', 'revenue']

        for col in required:
            if col not in df.columns:
                df[col] = 'Unknown'
            else:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()

        for col in optional:
            if col not in df.columns:
                df[col] = 'Unknown'
            else:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()

        df['year'] = pd.to_numeric(
            df['year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce'
        ).fillna(0).astype(int)

        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0).round(1)

        df = df.drop_duplicates(subset=['title', 'year'])
        df = df[df['title'] != 'Unknown']

        print(f'📊 Dataset: {len(df)} film | Tahun: {df[df["year"]>0]["year"].min()}–{df["year"].max()}')
        print(f'   Kolom tersedia: {list(df.columns)}')
        return df

    def to_documents(self, df: pd.DataFrame) -> List[Document]:
        documents = []
        for _, row in df.iterrows():
            content = (
                f"Title: {row['title']}\n"
                f"Original Title: {row.get('orig_title', 'Unknown')}\n"
                f"Year: {row['year']}\n"
                f"Genre: {row['genre']}\n"
                f"Crew: {row['crew']}\n"
                f"Cast: {row.get('cast', 'Unknown')}\n"
                f"Rating: {row['rating']}/10\n"
                f"Overview: {row['overview']}\n"
                f"Country: {row.get('country', 'Unknown')}\n"
                f"Language: {row.get('language', 'Unknown')}\n"
                f"Status: {row.get('status', 'Unknown')}\n"
                f"Budget: {row.get('budget', 'Unknown')}\n"
                f"Revenue: {row.get('revenue', 'Unknown')}"
            )
            metadata = {
                'title':      str(row['title']),
                'orig_title': str(row.get('orig_title', 'Unknown')),
                'year':       int(row['year']),
                'genre':      str(row['genre']).lower(),
                'rating':     float(row['rating']),
                'crew':       str(row['crew']),
                'director':   str(row['crew']),
                'cast':       str(row['crew']).lower(),
                'country':    str(row.get('country', 'Unknown')).lower(),
                'language':   str(row.get('language', 'Unknown')).lower(),
                'status':     str(row.get('status', 'Unknown')).lower(),
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents


class VectorStoreManager:
    """Mengelola ChromaDB vector store dengan embedding lokal (HuggingFace)."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        print('⏳ Memuat embedding model (download sekali ~30MB)...')
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f'✅ Embedding model siap: {EMBEDDING_MODEL}')
        self.vectorstore: Optional[Chroma] = None

    def build(self, documents: List[Document], batch_size: int = 100) -> None:
        print(f'⏳ Membangun vector store dari {len(documents)} dokumen...')
        self.vectorstore = Chroma.from_documents(
            documents=documents[:batch_size],
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name='movies'
        )
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            print(f'   Batch {i//batch_size + 1}: {min(i + batch_size, len(documents))}/{len(documents)}')
        print(f'✅ Vector store selesai dibangun di: {self.persist_dir}')

    def load(self) -> None:
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name='movies'
        )
        count = self.vectorstore._collection.count()
        print(f'✅ Vector store dimuat: {count} dokumen dari {self.persist_dir}')

    def build_or_load(self, documents: List[Document]) -> None:
        chroma_db_file = os.path.join(self.persist_dir, 'chroma.sqlite3')

        if os.path.exists(self.persist_dir) and os.path.exists(chroma_db_file):
            self.load()
            if self.vectorstore._collection.count() > 0:
                print(f'💡 Tip: set FORCE_REBUILD=True untuk rebuild ulang')
                return
            print('⚠️  Vector store kosong, rebuilding...')

        self.build(documents)


print('✅ MovieDataLoader & VectorStoreManager siap')

# ============================================================
# CELL 4 — MODULE 1: MEMORY & GUARDRAILS
# ============================================================


class ChatMemory:
    """Manajemen riwayat percakapan multi-turn."""

    def __init__(self, max_turns: int = MEMORY_WINDOW_TURNS):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns
        self.last_docs: List[Document] = []
        self.last_titles: List[str] = []

    def save_results(self, docs: List[Document]) -> None:
        self.last_docs = docs
        self.last_titles = [d.metadata.get('title', '') for d in docs]

    def get_last_docs(self) -> List[Document]:
        return self.last_docs

    def get_last_titles_context(self) -> str:
        if not self.last_titles:
            return ''
        lines = [f'{i+1}. {t}' for i, t in enumerate(self.last_titles)]
        return 'Film dari hasil pencarian sebelumnya:\n' + '\n'.join(lines)

    def add_user(self, message: str) -> None:
        self.history.append({'role': 'user', 'content': message})
        self._trim()

    def add_assistant(self, message: str) -> None:
        self.history.append({'role': 'assistant', 'content': message})

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_history(self) -> List[Dict[str, str]]:
        return self.history.copy()

    def get_recent_context(self, n_turns: int = 3) -> str:
        recent = self.history[-(n_turns * 2):]
        if not recent:
            return '(Tidak ada riwayat percakapan)'
        lines = []
        for msg in recent:
            role = 'User' if msg['role'] == 'user' else 'Bot'
            lines.append(f"{role}: {msg['content'][:250]}")
        return '\n'.join(lines)

    def clear(self) -> None:
        self.history = []

    def __len__(self) -> int:
        return len(self.history) // 2


class TopicGuardrail:
    """Guardrail berlapis untuk memastikan chatbot hanya menjawab topik film."""

    MOVIE_WHITELIST = {
        'halo', 'hai', 'hi', 'hello', 'hey',
        'selamat pagi', 'selamat siang', 'selamat malam', 'selamat sore',
        'apa kabar', 'gimana', 'hei',
        'terima kasih', 'makasih', 'thanks',
        'oke', 'ok', 'siap', 'boleh', 'bisa', 'film', 'movie', 'sinema',
        'bioskop', 'nonton', 'tonton',
        'rekomendasi', 'rekomendasikan', 'saran',
        'aktor', 'aktris', 'bintang', 'sutradara', 'pemain',
        'genre', 'alur', 'cerita', 'plot', 'ending', 'adegan',
        'horror', 'horor', 'action', 'thriller', 'comedy', 'komedi',
        'drama', 'romantis', 'animasi', 'dokumenter', 'sci-fi',
        'marvel', 'dc comics', 'disney', 'pixar', 'studio ghibli',
        'oscar', 'golden globe', 'imdb', 'rotten tomatoes',
        'sequel', 'prequel', 'remake', 'spin-off', 'franchise',
        'box office', 'trailer', 'cameo', 'karakter',
        'actor', 'actress', 'director', 'cinema', 'watch',
        'series', 'movie series', 'tv show', 'streaming',
        'screenplay', 'cinematography', 'soundtrack',
    }

    OFF_TOPIC_BLACKLIST = {
        'resep', 'masak', 'makanan', 'kuliner', 'restoran', 'catering',
        'politik', 'pemilu', 'pilpres', 'presiden', 'legislatif', 'korupsi',
        'coding', 'programming', 'source code', 'javascript',
        'database', 'sql', 'machine learning', 'neural network',
        'skincare', 'makeup', 'kosmetik', 'kecantikan', 'serum',
        'saham', 'investasi', 'kripto', 'crypto', 'bitcoin', 'forex',
        'deposito', 'tabungan', 'pinjaman',
        'liga sepakbola', 'pertandingan bola', 'skor bola',
        'obat', 'penyakit', 'dokter', 'resep dokter', 'gejala',
    }

    def __init__(self, llm_client):
        self.client = llm_client

    def check(self, query: str, history_context: str) -> Tuple[bool, str]:
        q = query.lower()

        if any(kw in q for kw in self.MOVIE_WHITELIST):
            return True, ''

        if any(kw in q for kw in self.OFF_TOPIC_BLACKLIST):
            return False, self._get_rejection_message()

        return self._llm_classify(query, history_context)

    def _llm_classify(self, query: str, history: str) -> Tuple[bool, str]:
        prompt = f"""Kamu adalah classifier topik untuk chatbot film.

Riwayat percakapan terakhir:
{history}

Query baru: "{query}"

Tugasmu: Apakah query ini relevan dengan topik FILM atau SINEMA?

PENTING: Pertimbangkan konteks!
- 'yang mana lebih bagus?' → RELEVAN jika konteks sebelumnya tentang film
- 'siapa pemerannya?' → RELEVAN (aktor film)
- 'berapa harganya?' → TIDAK relevan (harga tiket mungkin oke, tapi cek konteks)

Jawab hanya JSON:
{{"is_movie_related": true/false, "reason": "alasan singkat dalam 1 kalimat"}}"""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=80,
                temperature=0,
                response_format={'type': 'json_object'}
            )
            result = json.loads(response.choices[0].message.content)
            is_relevant = result.get('is_movie_related', True)
            print(f'   🔍 Guardrail LLM: {"ALLOW" if is_relevant else "REJECT"} — {result.get("reason", "")}')
            return (True, '') if is_relevant else (False, self._get_rejection_message())
        except Exception as e:
            print(f'   ⚠️ Guardrail LLM error (fail-open): {e}')
            return True, ''

    @staticmethod
    def _get_rejection_message() -> str:
        return (
            'Maaf, saya hanya bisa membantu tentang **film dan sinema**.\n\n'
            'Sepertinya pertanyaan Anda di luar topik tersebut. '
            'Saya siap membantu dengan:\n\n'
            '•**Rekomendasi film** — berdasarkan genre, mood, atau vibe\n'
            '•**Info film** — plot, cast, rating, sutradara\n'
            '•**Perbandingan** — mana film yang lebih bagus?\n'
            '•**Film by era** — film terbaik tahun tertentu\n\n'
            'Ada film apa yang ingin Anda tanyakan?'
        )


print('✅ ChatMemory & TopicGuardrail siap')

# ============================================================
# CELL 5 — MODULE 2: QUERY ROUTER
# ============================================================


@dataclass
class QueryIntent:
    """Hasil analisa query — menentukan strategi pencarian."""
    strategy: str
    filters: Dict[str, Any] = field(default_factory=dict)
    semantic_query: str = ''
    limit: int = DEFAULT_RESULT_LIMIT
    needs_external: bool = False
    raw_query: str = ''
    explicit_title: str = ''

    def __post_init__(self):
        self.limit = max(1, min(20, int(self.limit) if self.limit is not None else DEFAULT_RESULT_LIMIT))

    def summary(self) -> str:
        parts = [f'strategy={self.strategy}', f'limit={self.limit}']
        if self.filters:
            parts.append(f'filters={self.filters}')
        if self.needs_external:
            parts.append('needs_external=True')
        if self.explicit_title:
            parts.append(f'explicit_title="{self.explicit_title}"')
        return ' | '.join(parts)


class QueryRouter:
    """Menganalisa query user untuk menentukan strategi pencarian optimal."""

    ROUTING_PROMPT = """\
Kamu adalah query analyzer untuk sistem pencarian film.

Riwayat percakapan:
{history}

Query user: "{query}"

Analisa query dan hasilkan JSON dengan struktur berikut:

{{
  "strategy": "metadata_filter" | "semantic_search" | "hybrid",
  "filters": {{
    "genre": null atau string,
    "year_from": null atau integer,
    "year_to": null atau integer,
    "min_rating": null atau float,
    "actor": null atau string,
    "director": null atau string
  }},
  "semantic_query": "kalimat deskriptif panjang untuk similarity search",
  "limit": integer,
  "needs_external": boolean,
  "explicit_title": null atau string
}}

ATURAN STRATEGI:
- 'metadata_filter': query spesifik berdasarkan atribut → "film comedy rating di atas 8"
- 'semantic_search': berbasis vibe/plot → "film seperti Interstellar", "film tentang balas dendam"
- 'hybrid': ada filter DAN semantic → "film sci-fi yang mirip Interstellar"

ATURAN explicit_title (PENTING - baca dengan teliti):
- Isi jika user menyebut judul film secara spesifik
- Contoh: "crew dari film Bloody Hell" → explicit_title: "Bloody Hell"
- Contoh: "berapa budget Memory?" → explicit_title: "Memory"
- Contoh: "film nomor 3" → explicit_title: null (referensi ke list, bukan judul)
- TANDA KUTIP: Teks dalam tanda kutip tunggal/ganda ADALAH judul film!
  - "film 'agak laen' membahas apa" → explicit_title: "agak laen"
  - 'film "comic 8" itu tentang apa' → explicit_title: "comic 8"
- JUDUL INDONESIA: Ekstrak judul film Indonesia meski terdengar umum:
  - "film agak laen", "film comic 8", "film qorin 2", "film satans slaves"
  - → explicit_title wajib diisi
- QUERY SIMILARITY: "film mirip X", "film seperti X", "ada film yang serupa dengan X"
  - → explicit_title: "X" (judul film rujukan), strategy: "semantic_search"

ATURAN semantic_query (PENTING):
- Harus berupa deskripsi PANJANG dan KAYA konten dalam BAHASA INGGRIS
- Contoh BURUK: "film mirip Interstellar"
- Contoh BAGUS: "science fiction space exploration time relativity emotional journey stunning visuals"
- Untuk similarity query: deskripsikan PLOT/VIBE dari judul rujukan, BUKAN judulnya

ATURAN needs_external=true:
- Menyebut tahun 2024, 2025, 2026
- User bertanya 'film terbaru' tanpa batas tahun jelas

ATURAN limit:
- 'top 10', 'sepuluh film' → limit: 10
- 'beberapa' → limit: 5
- tidak disebutkan → limit: {default_limit}

Jawab HANYA dengan JSON valid tanpa penjelasan tambahan."""

    INDEX_REF_PATTERNS = [
        r'\bno\.?\s*\d+\b',
        r'\bnomor\s*\d+\b',
        r'\bfilm\s+(?:ke-?\d+|pertama|kedua|ketiga|keempat|kelima)\b',
        r'\byang\s+(?:pertama|kedua|ketiga|keempat|kelima)\b',
        r'\bdi\s+(?:list|urutan|daftar)\s+(?:nomor\s*)?\d+\b',
    ]

    GENERIC_FOLLOWUP_PATTERNS = [
        r'\bjelaskan\s+(?:lebih|detail)\b',
        r'\bceritakan\s+(?:tentang|lebih|apa)\b',
        r'\bbercerita\s+tentang\b',
        r'\binfo\s+(?:lebih|detail|lengkap)\b',
        r'\btayang\s+(?:kapan|tahun)\b',
        r'\btahun\s+berapa\b',
        r'\bkapan\s+tayang\b',
        r'\bberapa\s+(?:rating|tahun|durasi)\b',
        r'\bsiapa\s+(?:saja\s+)?(?:crew|sutradara|pemain|cast|aktor)\b',
        r'\bberapa\s+(?:revenue|budget|pendapatan)\b',
        r'\brevenue\b',
        r'\bbudget\b',
        r'\bpendapatan\b',
        r'\bcontext\s+(?:apa|yang)\b',
        r'\blist(?:nya)?\s+(?:apa|tadi)\b',
        r'\bapa\s+saja\s+(?:yang|film|tadi)\b',
        r'\btadi\s+(?:kan|kamu|ada)\b',
        r'\byang\s+(?:tadi|sebelumnya|kamu\s+(?:sebut|rekomen))\b',
        r'\bdalam\s+list\b',
        r'\bdari\s+(?:list|daftar)\s+(?:tadi|sebelumnya)\b',
		r'\blist\s+di\s+atas\b',
        r'\bdari\s+(?:list|daftar)\s+di\s+atas\b',
        r'\bmana\s+(?:yang\s+)?(?:paling|lebih)\b',
        r'\byang\s+(?:paling|lebih)\s+(?:direkomendasikan|bagus|baik|worth|recommended)\b',
        r'\bpaling\s+(?:direkomendasikan|bagus|worth|recommended|oke)\b',
        r'\brekomen(?:dasikan)?\s+(?:yang|dari|mana)\b',
        r'\bdi\s+atas\b',
    ]

    SIMILARITY_PATTERNS = [
        r'\bmirip\s+(?:dengan\s+)?\w',
        r'\bseperti\b',
        r'\bsimilar\s+to\b',
        r'\brekomendasi.{0,20}mirip\b',
        r'\bfilm\s+lain\s+yang\b',
        r'\byang\s+serupa\b',
        r'\bsejenisnya\b',
        r'\bsama\s+vibe\b',
        r'\bsama\s+genre\b',
        r'\bada\s+(?:film|movie)\s+(?:lain|lainnya)\b',
    ]

    TITLE_STOP_WORDS = {
        'the', 'a', 'an', 'in', 'of', 'and', 'or', 'to', 'is', 'are',
        'was', 'be', 'as', 'at', 'by', 'for', 'on', 'with', 'from',
    }

    def __init__(self, llm_client):
        self.client = llm_client

    def route(self, query: str, history_context: str) -> QueryIntent:
        prompt = self.ROUTING_PROMPT.format(
            history=history_context,
            query=query,
            default_limit=DEFAULT_RESULT_LIMIT
        )
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=300,
                temperature=0,
                response_format={'type': 'json_object'}
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)

            raw_limit = data.get('limit')
            safe_limit = int(raw_limit) if raw_limit is not None else DEFAULT_RESULT_LIMIT

            return QueryIntent(
                strategy=data.get('strategy', 'semantic_search'),
                filters=self._clean_filters(data.get('filters', {})),
                semantic_query=data.get('semantic_query', query),
                limit=safe_limit,
                needs_external=bool(data.get('needs_external', False)),
                raw_query=query,
                explicit_title=data.get('explicit_title') or '',
            )
        except Exception as e:
            print(f'   ⚠️ Router error, fallback semantic: {e}')
            needs_ext = bool(re.search(r'202[4-9]|203\d', query))
            return QueryIntent(
                strategy='semantic_search',
                semantic_query=query,
                needs_external=needs_ext,
                raw_query=query
            )

    @staticmethod
    def _clean_filters(filters: Dict) -> Dict:
        return {k: v for k, v in filters.items() if v is not None}

    def is_index_ref(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in self.INDEX_REF_PATTERNS)

    def is_generic_followup(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in self.GENERIC_FOLLOWUP_PATTERNS)

    def is_similarity_query(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in self.SIMILARITY_PATTERNS)

    def extract_referenced_index(self, query: str) -> Optional[int]:
        q = query.lower()
        m = re.search(r'(?:no\.?\s*|nomor\s*|ke-?)(\d+)', q)
        if m:
            return int(m.group(1)) - 1
        ordinals = {'pertama': 0, 'kedua': 1, 'ketiga': 2, 'keempat': 3, 'kelima': 4}
        for word, idx in ordinals.items():
            if word in q:
                return idx
        return None

    def find_title_in_cache(self, query: str, last_titles: List[str]) -> Optional[int]:
        if not last_titles:
            return None
        q = query.lower()
        for i, title in enumerate(last_titles):
            title_words = title.lower().split()
            if len(title_words) >= 2:
                for j in range(len(title_words) - 1):
                    bigram = f"{title_words[j]} {title_words[j+1]}"
                    if bigram in q:
                        return i
            elif len(title_words) == 1 and title_words[0] in q:
                return i
        return None

    def verify_title_match(self, explicit_title: str, found_title: str) -> bool:
        if not explicit_title or not found_title:
            return False
        exp_words = {
            w for w in explicit_title.lower().split()
            if len(w) > 2 and w not in self.TITLE_STOP_WORDS
        }
        if not exp_words:
            return explicit_title.lower().strip() in found_title.lower()
        found_lower = found_title.lower()
        return any(w in found_lower for w in exp_words)

    def resolve_query_mode(
        self,
        query: str,
        intent: QueryIntent,
        last_titles: List[str],
        last_docs_count: int,
    ) -> str:
        has_cache = last_docs_count > 0

        if has_cache and self.is_index_ref(query):
            ref_idx = self.extract_referenced_index(query)
            if ref_idx is not None and ref_idx < last_docs_count:
                return 'cache_index'

        if self.is_similarity_query(query):
            return 'similarity'

        if has_cache:
            lookup_key = intent.explicit_title if intent.explicit_title else query
            title_match = self.find_title_in_cache(lookup_key, last_titles)
            if title_match is not None:
                return 'cache_title'

        if intent.explicit_title:
            return 'db_title'

        if has_cache and self.is_generic_followup(query) and not intent.explicit_title:
            return 'cache_follow'

        if intent.needs_external:
            return 'external'

        return 'db_search'


print('✅ QueryIntent & QueryRouter siap (v3 — BUG 1,3,4 fixed)')

# ============================================================
# CELL 6 — MODULE 2: SEARCH ENGINE
# ============================================================


class MovieSearchEngine:
    """Mesin pencarian film dengan 3 strategi."""

    def __init__(self, vsm: VectorStoreManager):
        self.vsm = vsm

    def search(self, intent: QueryIntent) -> List[Document]:
        if intent.strategy == 'metadata_filter':
            return self._metadata_search(intent)
        elif intent.strategy == 'semantic_search':
            return self._semantic_search(intent)
        else:
            return self._hybrid_search(intent)

    def _metadata_search(self, intent: QueryIntent) -> List[Document]:
        candidate_limit = max(intent.limit * 5, 50)
        try:
            query = intent.semantic_query or intent.raw_query or 'popular movies'
            candidates = self.vsm.vectorstore.similarity_search(query, k=candidate_limit)

            if intent.filters:
                filtered = [
                    doc for doc in candidates
                    if self._passes_filter(doc.metadata, intent.filters)
                ]
                if len(filtered) < intent.limit:
                    print(f'   ⚠️ Filter ketat ({len(filtered)} hasil), melonggarkan...')
                    candidates = self.vsm.vectorstore.similarity_search(query, k=candidate_limit * 3)
                    filtered = [
                        doc for doc in candidates
                        if self._passes_filter(doc.metadata, intent.filters)
                    ]
            else:
                filtered = candidates

            filtered.sort(key=lambda d: d.metadata.get('rating', 0), reverse=True)
            return filtered[:intent.limit]
        except Exception as e:
            print(f'   ⚠️ Metadata search error: {e}')
            return self._semantic_search(intent)

    def _semantic_search(self, intent: QueryIntent) -> List[Document]:
        query = intent.semantic_query or intent.raw_query
        try:
            candidates = self.vsm.vectorstore.similarity_search(query, k=MAX_SEMANTIC_CANDIDATES)
            candidates.sort(key=lambda d: d.metadata.get('rating', 0), reverse=True)
            return candidates[:intent.limit]
        except Exception as e:
            print(f'   ⚠️ Semantic search error: {e}')
            return []

    def _hybrid_search(self, intent: QueryIntent) -> List[Document]:
        query = intent.semantic_query or intent.raw_query
        try:
            candidates = self.vsm.vectorstore.similarity_search(query, k=MAX_SEMANTIC_CANDIDATES * 2)
            filtered = [doc for doc in candidates if self._passes_filter(doc.metadata, intent.filters)]
            filtered.sort(key=lambda d: d.metadata.get('rating', 0), reverse=True)

            if len(filtered) < intent.limit // 2:
                print('   ⚠️ Hybrid: hasil terlalu sedikit, fallback ke semantic')
                return self._semantic_search(intent)

            return filtered[:intent.limit]
        except Exception as e:
            print(f'   ⚠️ Hybrid search error: {e}')
            return self._semantic_search(intent)

    @staticmethod
    def _build_where_clause(filters: Dict) -> Optional[Dict]:
        conditions = []
        if filters.get('genre'):
            conditions.append({'genre': {'$contains': filters['genre'].lower()}})
        if filters.get('min_rating') is not None:
            conditions.append({'rating': {'$gte': float(filters['min_rating'])}})
        if filters.get('year_from') is not None:
            conditions.append({'year': {'$gte': int(filters['year_from'])}})
        if filters.get('year_to') is not None:
            conditions.append({'year': {'$lte': int(filters['year_to'])}})
        if filters.get('actor'):
            conditions.append({'cast': {'$contains': filters['actor'].lower()}})
        if filters.get('director'):
            conditions.append({'director': {'$contains': filters['director'].lower()}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {'$and': conditions}

    @staticmethod
    def _passes_filter(metadata: Dict, filters: Dict) -> bool:
        genre_meta = str(metadata.get('genre', '')).lower()
        if filters.get('genre'):
            requested_genres = [g.strip() for g in re.split(r'[\s,|/]+', filters['genre'].lower()) if g.strip()]
            if not all(g in genre_meta for g in requested_genres):
                return False
        if filters.get('min_rating') is not None:
            if metadata.get('rating', 0) < filters['min_rating']:
                return False
        if filters.get('year_from') is not None:
            if metadata.get('year', 0) < filters['year_from']:
                return False
        if filters.get('year_to') is not None:
            if metadata.get('year', 9999) > filters['year_to']:
                return False
        if filters.get('actor'):
            if filters['actor'].lower() not in str(metadata.get('cast', '')).lower():
                return False
        if filters.get('director'):
            if filters['director'].lower() not in str(metadata.get('director', '')).lower():
                return False
        return True


print('✅ MovieSearchEngine siap (metadata | semantic | hybrid)')

# ============================================================
# CELL 7 — MODULE 3: FALLBACK EXTERNAL API
# ============================================================


class ExternalMovieAPI:
    """Fallback ke TMDB / OMDb untuk film yang tidak ada di database lokal."""

    TMDB_BASE = 'https://api.themoviedb.org/3'
    OMDB_BASE = 'http://www.omdbapi.com'
    TIMEOUT   = 10

    def __init__(self, tmdb_key: str = '', omdb_key: str = ''):
        self.tmdb_key = tmdb_key
        self.omdb_key = omdb_key

    def is_available(self) -> bool:
        return bool(self.tmdb_key or self.omdb_key)

    def search(self, query: str, intent: QueryIntent) -> Tuple[List[Dict], str]:
        if self.tmdb_key:
            results = self._tmdb_search(query, intent)
            if results:
                return results, 'TMDB (The Movie Database)'
        if self.omdb_key:
            results = self._omdb_search(query, intent)
            if results:
                return results, 'OMDb (Open Movie Database)'
        return [], 'Tidak ada sumber eksternal tersedia'

    def _tmdb_search(self, query: str, intent: QueryIntent) -> List[Dict]:
        try:
            params = {
                'api_key': self.tmdb_key,
                'query': query,
                'language': 'id-ID',
                'page': 1,
                'include_adult': 'false'
            }
            if intent.filters.get('year_from'):
                params['primary_release_date.gte'] = f"{intent.filters['year_from']}-01-01"
            if intent.filters.get('year_to'):
                params['primary_release_date.lte'] = f"{intent.filters['year_to']}-12-31"

            resp = requests.get(f'{self.TMDB_BASE}/search/movie', params=params, timeout=self.TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            movies = []
            for m in data.get('results', [])[:intent.limit]:
                year = m.get('release_date', '')[:4] or 'Unknown'
                movies.append({
                    'title':    m.get('title', 'Unknown'),
                    'year':     year,
                    'rating':   round(m.get('vote_average', 0), 1),
                    'overview': m.get('overview', 'Deskripsi tidak tersedia')[:300],
                    'genre':    'Lihat TMDB',
                    'source':   'TMDB'
                })
            return movies
        except requests.Timeout:
            print('   ⚠️ TMDB timeout')
        except requests.HTTPError as e:
            print(f'   ⚠️ TMDB HTTP error: {e}')
        except Exception as e:
            print(f'   ⚠️ TMDB error: {e}')
        return []

    def _omdb_search(self, query: str, intent: 'QueryIntent') -> List[Dict]:
        try:
            resp = requests.get(
                self.OMDB_BASE,
                params={'apikey': self.omdb_key, 's': query, 'type': 'movie'},
                timeout=self.TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get('Response') == 'False':
                return []
            movies = []
            for item in data.get('Search', [])[:intent.limit]:
                imdb_id = item.get('imdbID', '')
                detail = self._omdb_detail(imdb_id)
                if detail:
                    movies.append(detail)
            return movies
        except Exception as e:
            print(f'   ⚠️ OMDb error: {e}')
        return []

    def _omdb_detail(self, imdb_id: str) -> Optional[Dict]:
        if not imdb_id:
            return None
        try:
            resp = requests.get(
                self.OMDB_BASE,
                params={'apikey': self.omdb_key, 'i': imdb_id, 'plot': 'short'},
                timeout=5
            )
            d = resp.json()
            if d.get('Response') == 'False':
                return None
            return {
                'title':    d.get('Title', 'Unknown'),
                'year':     d.get('Year', 'Unknown'),
                'rating':   d.get('imdbRating', 'N/A'),
                'genre':    d.get('Genre', 'Unknown'),
                'overview': d.get('Plot', 'No description'),
                'director': d.get('Director', 'Unknown'),
                'cast':     d.get('Actors', 'Unknown'),
                'source':   'OMDb'
            }
        except Exception as e:
            print(f'   ⚠️ OMDb detail error: {e}')
            return None

    @staticmethod
    def format_results(movies: List[Dict]) -> str:
        if not movies:
            return ''
        lines = []
        for i, m in enumerate(movies, 1):
            lines.append(f"{i}. Title: {m.get('title', 'Unknown')}")
            lines.append(f"   Year: {m.get('year', 'Unknown')}")
            lines.append(f"   Rating: {m.get('rating', 'N/A')} | Genre: {m.get('genre', 'N/A')}")
            overview = str(m.get('overview', ''))[:250]
            lines.append(f"   Overview: {overview}")
            if m.get('director') and m['director'] not in ('Unknown', 'Lihat TMDB'):
                lines.append(f"   Director: {m['director']}")
            lines.append('')
        return '\n'.join(lines)


print('✅ ExternalMovieAPI siap (TMDB + OMDb + sitasi)')

# ============================================================
# CELL 8 — RESPONSE GENERATOR
# ============================================================


class ResponseGenerator:
    """Menghasilkan jawaban natural dari context film menggunakan LLM."""

    SYSTEM_PROMPT = """\
Kamu adalah FilmAI, asisten AI yang ramah dan berpengetahuan luas tentang film.

KEPRIBADIAN:
- Antusias saat membahas film
- Berbicara dalam Bahasa Indonesia yang natural
- Berikan jawaban yang informatif namun ringkas
- Jika user hanya menyapa (halo, hai, hi, selamat pagi, dll) tanpa pertanyaan film:
  balas sapaan dengan ramah, perkenalkan dirimu sebagai FilmAI, dan TANYA apa yang ingin mereka cari.
  JANGAN langsung rekomendasikan film tanpa diminta.

ATURAN SAPAAN (WAJIB):
- Jika user HANYA mengirim sapaan tanpa pertanyaan (contoh: "halo", "hai", "hi", "selamat pagi"):
  1. Balas sapaan dengan hangat
  2. Perkenalkan diri sebagai FilmAI
  3. Tanya apa yang ingin mereka cari atau tonton
  4. STOP — JANGAN rekomendasikan film apapun
  5. JANGAN tulis daftar film
  6. JANGAN tulis "Berikut beberapa film..."
- Contoh respons yang BENAR untuk sapaan:
  "Halo! Saya FilmAI, asisten film kamu. Mau cari film apa hari ini? Atau butuh rekomendasi genre tertentu?"
- Contoh respons yang SALAH untuk sapaan:
  "Halo! Berikut beberapa film menarik: 1. ..."

ATURAN KETAT (ANTI-HALUSINASI):
1. HANYA gunakan informasi dari data yang diberikan
2. JANGAN mengarang detail film (tanggal, cast, plot) yang tidak ada di data
3. Jika data kosong atau bertuliskan "TIDAK ADA DATA": sampaikan film tidak ditemukan
4. JANGAN menyebutkan bahwa kamu 'mencari' atau 'menggunakan database'

ATURAN FORMAT (WAJIB):
- JANGAN PERNAH menulis "[CONTEXT]", "[/CONTEXT]", atau kata "CONTEXT" dalam jawabanmu
- JANGAN sebut "berdasarkan context", "di context", "tidak ada di context", atau sejenisnya
- Jika informasi tidak tersedia, katakan: "Maaf, informasi ini tidak tersedia di database kami."
- Jawab natural seolah kamu memang tahu informasinya

ATURAN URUTAN TAMPILAN (WAJIB DIIKUTI):
- Tampilkan film PERSIS dalam urutan yang sama dengan urutan di data yang diberikan
- Film pertama = Nomor 1, film kedua = Nomor 2, dan seterusnya
- DILARANG KERAS mengubah, membalik, atau menyusun ulang urutan film

ATURAN JAWABAN BERDASARKAN JUMLAH DATA:
- Jika data hanya berisi 1 film: jawab pertanyaan user khusus tentang film itu
- Jika data berisi banyak film: tampilkan sesuai permintaan user
- Jika data kosong: sampaikan film tidak ditemukan, JANGAN tawarkan film lain

ATURAN SITASI (WAJIB):
- Data dari Database Lokal → cantumkan di akhir: "\\n\\n---\\n📡 Sumber: Database Lokal"
- Data dari API eksternal → cantumkan: "\\n\\n---\\n📡 Sumber: [nama API]"

FORMAT REKOMENDASI:
🎬 **[Judul]** ([Tahun]) — ⭐ [Rating]/10
[Penjelasan singkat kenapa film ini cocok dengan permintaan user]"""

    def __init__(self, llm_client):
        self.client = llm_client

    def generate(
        self,
        query: str,
        local_docs: List[Document],
        external_data: Optional[Tuple[List[Dict], str]],
        history: List[Dict],
        intent: QueryIntent,
        query_mode: str = 'db_search',
    ) -> str:
        context_sections = []
        has_external = False
        external_source = ''
        is_local = False

        if local_docs:
            local_text = '\n---\n'.join(doc.page_content for doc in local_docs)
            context_sections.append(
                f'=== DATA DARI DATABASE LOKAL ({len(local_docs)} film) ===\n{local_text}'
            )
            is_local = True

        if external_data and external_data[0]:
            ext_movies, ext_source = external_data
            ext_text = ExternalMovieAPI.format_results(ext_movies)
            context_sections.append(
                f'=== DATA DARI SUMBER EKSTERNAL: {ext_source} ===\n{ext_text}'
            )
            has_external = True
            external_source = ext_source

        context = '\n\n'.join(context_sections) if context_sections else (
            'TIDAK ADA DATA FILM yang ditemukan untuk query ini.'
        )

        messages = [{'role': 'system', 'content': self.SYSTEM_PROMPT}]
        messages.extend(history[-6:])

        if has_external:
            citation_instruction = (
                f'\nPENTING: Data dari "{external_source}" — '
                f'cantumkan "📡 Sumber: {external_source}" di akhir jawaban.'
            )
        elif is_local:
            citation_instruction = '\nPENTING: Data dari Database Lokal — cantumkan "📡 Sumber: Database Lokal" di akhir jawaban.'
        else:
            citation_instruction = ''

        n_docs = len(local_docs)
        if query_mode in ('cache_index', 'cache_title', 'db_title'):
            limit_instruction = (
                '- Jawab pertanyaan user tentang film yang ada di data\n'
                '- Fokus ke informasi yang relevan dengan pertanyaan\n'
                '- JANGAN katakan "data tidak mencukupi" — gunakan semua info yang tersedia'
            )
        elif query_mode == 'cache_follow':
            limit_instruction = (
                f'- Jawab pertanyaan user berdasarkan {n_docs} film di data\n'
                '- Gunakan semua informasi yang relevan dengan pertanyaan user'
            )
        else:
            limit_instruction = (
                f'- Tampilkan maksimal {intent.limit} film\n'
                f'- WAJIB tampilkan film dalam urutan PERSIS sama dengan urutan di data\n'
                f'  Film ke-1 di data = Nomor 1, film ke-2 = Nomor 2, dst. JANGAN ubah urutan!\n'
                '- JANGAN isi slot yang kosong dengan film fiktif atau tidak relevan'
            )

        user_content = (
            f'[DATA FILM]\n{context}\n[/DATA FILM]\n\n'
            f'Pertanyaan user: {query}\n\n'
            f'Instruksi:\n'
            f'{limit_instruction}'
            f'{citation_instruction}'
            f'\nPENTING: Jangan pernah menyebut "[DATA FILM]", "[/DATA FILM]", '
            f'"[CONTEXT]", atau istilah teknis internal lainnya dalam jawabanmu.'
        )

        messages.append({'role': 'user', 'content': user_content})

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=1800,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f'❌ Maaf, terjadi kesalahan saat memproses jawaban: {str(e)}'


print('✅ ResponseGenerator siap (v3 — BUG 2 ordering fixed + sitasi lokal)')

# ============================================================
# CELL 9 — MAIN ORCHESTRATOR: MovieRAGChatbot
# ============================================================

from groq import Groq


class MovieRAGChatbot:
    """Orkestrator utama yang menggabungkan semua modul."""

    GREETING_PATTERNS = {
        'halo', 'hai', 'hi', 'hello', 'hey', 'hei',
        'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam',
        'assalamu', 'permisi', 'hallo',
    }

    GREETING_RESPONSE = (
        "Halo! Saya **FilmAI**, asisten film kamu.\n"
        "Saya bisa membantu kamu untuk mencari film berdasarkan genre atau tahun."
		" Saya juga bisa memberikan info detail tentang film tertentu."
        " Mau cari atau tanya film apa hari ini?"
    )

    def _is_greeting_only(self, text: str) -> bool:
        """Cek apakah pesan hanya sapaan tanpa pertanyaan film."""
        t = text.lower().strip().rstrip('!.,?')
        if t in self.GREETING_PATTERNS:
            return True
        if any(t.startswith(g) for g in self.GREETING_PATTERNS) and len(t.split()) <= 4:
            return True
        return False

    def __init__(
        self,
        vsm: VectorStoreManager,
        groq_api_key: str = GROQ_API_KEY,
        tmdb_api_key: str = TMDB_API_KEY,
        omdb_api_key: str = OMDB_API_KEY,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.llm = Groq(api_key=groq_api_key)

        self.memory        = ChatMemory(max_turns=MEMORY_WINDOW_TURNS)
        self.guardrail     = TopicGuardrail(self.llm)
        self.router        = QueryRouter(self.llm)
        self.search_engine = MovieSearchEngine(vsm)
        self.external_api  = ExternalMovieAPI(tmdb_api_key, omdb_api_key)
        self.response_gen  = ResponseGenerator(self.llm)

        self._log('🎬 MovieRAGChatbot berhasil diinisialisasi!')
        self._log(f'   LLM         : {LLM_MODEL} via Groq')
        self._log(f'   Embedding   : {EMBEDDING_MODEL} (lokal)')
        self._log(f'   Memory      : {MEMORY_WINDOW_TURNS} turns')
        self._log(f'   External API: {"TMDB✅" if tmdb_api_key else "TMDB❌"} | {"OMDb✅" if omdb_api_key else "OMDb❌"}')

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def chat(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return '❓ Silakan ketikkan pertanyaan Anda.'

        self._log(f'\n{"─"*55}')
        self._log(f'👤 User: {user_input}')
        self._log(f'{"─"*55}')

        # CEK SAPAAN DULU sebelum apapun
        if self._is_greeting_only(user_input):
            self._log('       → Sapaan terdeteksi, skip ke greeting response')
            self.memory.add_user(user_input)
            self.memory.add_assistant(self.GREETING_RESPONSE)
            return self.GREETING_RESPONSE

        history_ctx = self.memory.get_recent_context(n_turns=3)

        # STEP 1: Guardrail
        self._log('[1/5] 🛡️  Guardrail check...')
        is_ok, rejection = self.guardrail.check(user_input, history_ctx)
        if not is_ok:
            self._log('       → DITOLAK (off-topic)')
            self.memory.add_user(user_input)
            self.memory.add_assistant(rejection)
            return rejection
        self._log('       → OK')

        # STEP 2: Query Routing
        self._log('[2/5] 🧭  Query routing...')
        intent = self.router.route(user_input, history_ctx)
        self._log(f'       → {intent.summary()}')

        # STEP 2.5: Resolve mode
        last_titles     = self.memory.last_titles
        last_docs       = self.memory.get_last_docs()
        last_docs_count = len(last_docs)

        mode = self.router.resolve_query_mode(
            query=user_input,
            intent=intent,
            last_titles=last_titles,
            last_docs_count=last_docs_count,
        )
        self._log(f'[2.5] 🔀  Mode: {mode}')

        # STEP 3: Eksekusi berdasarkan mode
        local_docs: List[Document] = []
        update_cache = False

        if mode == 'cache_index':
            ref_idx = self.router.extract_referenced_index(user_input)
            local_docs = [last_docs[ref_idx]]
            self._log(f'[3/5] 📚  Cache index [{ref_idx}]: "{last_titles[ref_idx]}"')

        elif mode == 'cache_title':
            lookup_key = intent.explicit_title if intent.explicit_title else user_input
            title_match = self.router.find_title_in_cache(lookup_key, last_titles)
            local_docs = [last_docs[title_match]]
            self._log(f'[3/5] 📚  Cache title: "{last_titles[title_match]}"')

        elif mode == 'cache_follow':
            local_docs = last_docs
            self._log(f'[3/5] 📚  Cache follow: {len(local_docs)} film dari list sebelumnya')

        elif mode == 'db_title':
            title_query = intent.explicit_title
            self._log(f'[3/5] 📚  DB title search: "{title_query}"')

            title_intent = QueryIntent(
                strategy='semantic_search',
                semantic_query=f'movie film titled {title_query} {title_query}',
                limit=3,
                raw_query=title_query,
            )
            candidates = self.search_engine.search(title_intent)

            local_docs = []
            for doc in candidates:
                found_title = doc.metadata.get('title', '')
                if self.router.verify_title_match(title_query, found_title):
                    local_docs = [doc]
                    self._log(f'       → 1 film cocok: "{found_title}"')
                    break

            if not local_docs:
                self._log(f'       → 0 film cocok (judul "{title_query}" tidak ada di DB — akan coba fallback)')

        elif mode == 'similarity':
            self._log(f'[3/5] 📚  Similarity search (semantic baru — cari film LAIN yang mirip)...')
            raw_results = self.search_engine.search(intent)

            if intent.explicit_title:
                source_words = {
                    w for w in intent.explicit_title.lower().split()
                    if len(w) > 3 and w not in self.router.TITLE_STOP_WORDS
                }
                local_docs = [
                    d for d in raw_results
                    if not any(
                        w in d.metadata.get('title', '').lower()
                        for w in source_words
                    )
                ] if source_words else raw_results
            else:
                local_docs = raw_results

            self._log(f'       → {len(local_docs)} film ditemukan (sumber dihapus dari hasil)')
            update_cache = True

        elif mode == 'external':
            self._log('[3/5] 📚  Skip local search (needs_external=True)')

        else:  # db_search
            self._log('[3/5] 📚  DB search...')
            local_docs = self.search_engine.search(intent)
            self._log(f'       → {len(local_docs)} film ditemukan')
            update_cache = True

        if update_cache and local_docs:
            self.memory.save_results(local_docs)
            self._log(f'       → Cache diperbarui ({len(local_docs)} film)')

        # STEP 4: Fallback External API
        external_data = None
        should_fallback = (
            mode == 'external'
            or (mode == 'db_title' and len(local_docs) == 0)
            or (mode == 'db_search' and len(local_docs) == 0)
        )

        if should_fallback and self.external_api.is_available():
            self._log('[4/5] 🌐  Fetching external API...')
            ext_query = intent.explicit_title or intent.semantic_query or user_input
            external_data = self.external_api.search(ext_query, intent)
            n_ext = len(external_data[0]) if external_data else 0
            src   = external_data[1] if external_data else '-'
            self._log(f'       → {n_ext} film dari {src}')
        elif should_fallback:
            self._log('[4/5] 🌐  External API tidak tersedia (API key belum diset)')
        else:
            self._log('[4/5] 🌐  Fallback tidak diperlukan')

        # STEP 5: Generate Response
        self._log('[5/5] 💬  Generating response...')
        response = self.response_gen.generate(
            query=user_input,
            local_docs=local_docs,
            external_data=external_data,
            history=self.memory.get_history(),
            intent=intent,
            query_mode=mode,
        )

        self.memory.add_user(user_input)
        self.memory.add_assistant(response)
        self._log(f'   Memory: {len(self.memory)} turns tersimpan')

        return response

    def reset(self) -> None:
        self.memory.clear()
        self._log('🔄 Memory percakapan direset')

    def get_history(self) -> List[Dict]:
        return self.memory.get_history()


print('✅ MovieRAGChatbot siap (v3 — BUG 1,3,5 fixed)')

# ============================================================
# CELL 10 — INISIALISASI SISTEM
# ============================================================

print('\n📂 Loading dataset...')
loader = MovieDataLoader()
df = loader.load(MOVIE_CSV_PATH)
docs = loader.to_documents(df)
print(f'   {len(docs)} dokumen siap untuk di-embed\n')

print('🗄️ Menyiapkan Vector Store...')
vsm = VectorStoreManager(persist_dir=CHROMA_PERSIST_DIR)
vsm.build_or_load(docs)
print()

print('🤖 Menginisialisasi MovieRAGChatbot...')
chatbot = MovieRAGChatbot(
    vsm=vsm,
    groq_api_key=GROQ_API_KEY,
    omdb_api_key=OMDB_API_KEY,
    verbose=True
)
print()
print('🎬 Sistem siap!')

# ============================================================
# FLASK APP — API ENDPOINTS untuk index.html
# ============================================================

app = Flask(__name__)

sessions: Dict[str, MovieRAGChatbot] = {}


def get_or_create_session(session_id: Optional[str]) -> Tuple[str, MovieRAGChatbot]:
    """Ambil atau buat chatbot instance per session."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = str(uuid.uuid4())
    new_chatbot = MovieRAGChatbot(
        vsm=vsm,
        groq_api_key=GROQ_API_KEY,
        omdb_api_key=OMDB_API_KEY,
        verbose=True
    )
    sessions[new_id] = new_chatbot
    return new_id, new_chatbot


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    message    = (data.get('message') or '').strip()
    session_id = data.get('session_id')

    if not message:
        return jsonify({'error': 'Pesan tidak boleh kosong'}), 400

    sid, bot = get_or_create_session(session_id)

    try:
        answer = bot.chat(message)
        movies_retrieved = bot.memory.last_titles[:10]
        return jsonify({
            'answer':           answer,
            'session_id':       sid,
            'movies_retrieved': movies_retrieved,
        })
    except Exception as e:
        print(f'[ERROR] /api/chat: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def api_reset():
    data       = request.get_json(force=True)
    session_id = data.get('session_id')

    if session_id and session_id in sessions:
        sessions[session_id].reset()

    return jsonify({'status': 'ok'})


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)