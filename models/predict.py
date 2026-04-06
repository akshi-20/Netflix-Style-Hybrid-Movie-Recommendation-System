import os
import pickle
import numpy as np
import pandas as pd


# =====================================
# PATH CONFIG
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")


# =====================================
# LOAD ARTIFACTS ONCE (CRITICAL FIX)
# =====================================

print("🚀 Loading recommendation artifacts...")

with open(os.path.join(MODELS_PATH, "topk_similarities.pkl"), "rb") as f:
    TOPK_DICT = pickle.load(f)

with open(os.path.join(MODELS_PATH, "tfidf_matrix.pkl"), "rb") as f:
    TFIDF_MATRIX = pickle.load(f)

METADATA = pd.read_parquet(
    os.path.join(MODELS_PATH, "movie_metadata.parquet")
)

FULL_DATA = pd.read_parquet(
    os.path.join(PROCESSED_PATH, "movies_final.parquet")
)

print("✅ Artifacts loaded successfully.")
# print(METADATA.columns)
# print(FULL_DATA.columns)


# =====================================
# PRECOMPUTE SCORES (Huge Speed Boost)
# =====================================

# Popular score
METADATA["pop_score"] = (
    METADATA["rating"] * np.log1p(METADATA["votes"])
)

POPULAR_SORTED = METADATA.sort_values(
    "pop_score",
    ascending=False
)

# Trending score (recent only)
RECENT = METADATA[METADATA["year"] >= 2018].copy()
RECENT["trend_score"] = (
    RECENT["rating"] * np.log1p(RECENT["votes"])
)

TRENDING_SORTED = RECENT.sort_values(
    "trend_score",
    ascending=False
)

print("🔥 Precomputed rankings ready.")


# =====================================
# SMART TITLE MATCH
# =====================================

def get_movie_index(title):
    title_clean = title.strip().lower()

    exact_match = METADATA[
        METADATA["title"].str.lower() == title_clean
    ]

    if not exact_match.empty:
        return exact_match.index[0]

    partial_match = METADATA[
        METADATA["title"].str.lower().str.contains(title_clean)
    ]

    if not partial_match.empty:
        return partial_match.index[0]

    return None


# =====================================
# HYBRID RANKING
# =====================================

def apply_hybrid_ranking(df):

    df = df.copy()

    df["rating_norm"] = df["rating"] / 10

    df["votes_norm"] = (
        np.log1p(df["votes"]) /
        np.log1p(df["votes"].max())
    )

    df["final_score"] = (
        0.6 * df["similarity"] +
        0.25 * df["rating_norm"] +
        0.15 * df["votes_norm"]
    )

    return df.sort_values("final_score", ascending=False)


# =====================================
# 1️⃣ BECAUSE YOU WATCHED
# =====================================

def recommend_similar(title, top_n=10):

    idx = get_movie_index(title)

    if idx is None:
        return "Movie not found."

    similar_items = TOPK_DICT.get(idx, [])

    if not similar_items:
        return "No recommendations found."

    indices = [item[0] for item in similar_items]
    similarities = [item[1] for item in similar_items]

    recs = METADATA.iloc[indices].copy()
    recs["similarity"] = similarities

    recs = apply_hybrid_ranking(recs)

    return recs.head(top_n).reset_index(drop=True)


# =====================================
# 2️⃣ USER PROFILE RECOMMENDER
# =====================================

def recommend_for_user(watched_titles, top_n=15):

    watched_indices = []

    for title in watched_titles:
        idx = get_movie_index(title)
        if idx is not None:
            watched_indices.append(idx)

    if not watched_indices:
        return "No valid watched movies found."

    user_vector = TFIDF_MATRIX[watched_indices].mean(axis=0)

    sim_scores = user_vector.dot(TFIDF_MATRIX.T).A1

    sim_scores[watched_indices] = 0

    top_indices = np.argsort(sim_scores)[::-1][:top_n]

    recs = METADATA.iloc[top_indices].copy()
    recs["similarity"] = sim_scores[top_indices]

    recs = apply_hybrid_ranking(recs)

    return recs.head(top_n).reset_index(drop=True)


# =====================================
# 3️⃣ POPULAR MOVIES (Instant Now)
# =====================================

def popular_movies(top_n=10):
    return POPULAR_SORTED.head(top_n).reset_index(drop=True)


# =====================================
# 4️⃣ TRENDING MOVIES (Instant Now)
# =====================================

def trending_movies(top_n=10):
    return TRENDING_SORTED.head(top_n).reset_index(drop=True)


# =====================================
# 5️⃣ GENRE ROW (Optimized)
# =====================================

def recommend_by_genre(genre, top_n=10):

    filtered = FULL_DATA[
        FULL_DATA["genres"].str.contains(genre, case=False)
    ].copy()

    filtered["score"] = (
        filtered["rating"] * np.log1p(filtered["votes"])
    )

    return filtered.sort_values(
        "score",
        ascending=False
    ).head(top_n).reset_index(drop=True)

# =====================================
# CLI TEST MODE (Optional)
# =====================================

if __name__ == "__main__":

    print("\n=== NETFLIX STYLE RECOMMENDER TEST MODE ===\n")

    while True:
        movie = input("Enter a movie (or 'exit'): ").strip()

        if movie.lower() == "exit":
            break

        print("\n--- Because You Watched ---\n")
        result = recommend_similar(movie)

        if isinstance(result, str):
            print(result)
        else:
            print(result[["title", "rating", "year"]])

        print("\n--- Popular Right Now ---\n")
        print(popular_movies(5)[["title", "rating"]])

        print("\n--- Trending Movies ---\n")
        print(trending_movies(5)[["title", "rating"]])
        print("\n" + "="*50 + "\n")