"""
RAG Engine using TF-IDF + cosine similarity (no internet needed).
Optimized for movie search: title, genre, overview, crew.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Load & clean dataset ────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    str_cols = ['names', 'date_x', 'genre', 'overview', 'crew', 'orig_title', 'status', 'orig_lang', 'country']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df['genre']    = df['genre'].fillna('Unknown')
    df['crew']     = df['crew'].fillna('Unknown')
    df['overview'] = df['overview'].fillna('No overview available.')
    df = df.drop_duplicates(subset='names').reset_index(drop=True)
    df = df[df['overview'].str.len() > 20].reset_index(drop=True)
    df['score']    = pd.to_numeric(df['score'],    errors='coerce').fillna(0)
    df['budget_x'] = pd.to_numeric(df['budget_x'], errors='coerce').fillna(0)
    df['revenue']  = pd.to_numeric(df['revenue'],  errors='coerce').fillna(0)
    return df


def fmt_money(val: float) -> str:
    if val >= 1e9:   return f"${val/1e9:.2f}B"
    elif val >= 1e6: return f"${val/1e6:.1f}M"
    elif val > 0:    return f"${val:,.0f}"
    return "N/A"


def build_searchable_doc(row) -> str:
    crew_short = ', '.join(row['crew'].split(',')[:8])
    return (
        f"{row['names']} {row['orig_title']} "
        f"genre {row['genre']} "
        f"{row['overview']} "
        f"cast {crew_short} "
        f"language {row['orig_lang']} "
        f"country {row['country']} "
        f"year {row['date_x']}"
    ).lower()


def row_to_dict(row) -> dict:
    return {
        "title":      row['names'],
        "orig_title": row['orig_title'],
        "release":    row['date_x'],
        "genre":      row['genre'],
        "score":      float(row['score']),
        "status":     row['status'],
        "language":   row['orig_lang'],
        "country":    row['country'],
        "budget":     fmt_money(row['budget_x']),
        "revenue":    fmt_money(row['revenue']),
        "crew":       row['crew'],
        "overview":   row['overview'],
    }


# ── Global state ────────────────────────────────────────────────────────────────
_df: pd.DataFrame = None
_vectorizer: TfidfVectorizer = None
_matrix = None


def init_engine(csv_path: str):
    global _df, _vectorizer, _matrix
    print("Loading dataset...")
    _df = load_dataset(csv_path)
    print(f"  {len(_df)} movies loaded.")

    print("Building TF-IDF index...")
    docs = [build_searchable_doc(row) for _, row in _df.iterrows()]
    _vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=60000,
        sublinear_tf=True,
    )
    _matrix = _vectorizer.fit_transform(docs)
    print(f"  TF-IDF matrix: {_matrix.shape}")


def retrieve_movies(query: str, top_k: int = 7, threshold: float = 0.05) -> list:
    q_vec = _vectorizer.transform([query.lower()])
    sims = cosine_similarity(q_vec, _matrix).flatten()
    idxs = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in idxs:
        score = float(sims[idx])
        if score >= threshold:
            d = row_to_dict(_df.iloc[idx])
            d['_sim'] = score
            results.append(d)
    return results


def get_movies_by_titles(titles: list) -> list:
    results = []
    for title in titles:
        title_clean = title.strip().lower()
        matches = _df[_df['names'].str.lower() == title_clean]
        if matches.empty:
            matches = _df[_df['names'].str.lower().str.contains(re.escape(title_clean), na=False)]
        if not matches.empty:
            results.append(row_to_dict(matches.iloc[0]))
    return results


def expand_query(query: str, history: list) -> str:
    if not history or len(query.split()) > 12:
        return query
    recent_msgs = [m['content'] for m in history[-4:] if m['role'] == 'user']
    keywords = []
    for msg in recent_msgs:
        words = [w for w in re.split(r'\W+', msg) if len(w) > 3]
        keywords.extend(words[:6])
    if not keywords:
        return query
    return f"{query} {' '.join(dict.fromkeys(keywords))}"
