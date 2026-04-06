print("TRAINING STARTED", flush=True)

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# =====================================
# PATH CONFIG
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_PATH = os.path.join(PROJECT_ROOT, "artifacts")

os.makedirs(MODELS_PATH, exist_ok=True)

TOP_K = 20


# =====================================
# LOAD FEATURES
# =====================================

def load_features():

    path = os.path.join(PROCESSED_PATH, "movies_features.parquet")

    print(f"Loading features from: {path}", flush=True)

    df = pd.read_parquet(path)
    df["tags"] = df["tags"].fillna("")

    print(f"Loaded {len(df)} movies", flush=True)

    return df


# =====================================
# BUILD TF-IDF MATRIX
# =====================================

def build_tfidf(df):

    print("Building TF-IDF matrix...", flush=True)

    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(df["tags"])

    # Normalize for cosine similarity
    tfidf_matrix = normalize(tfidf_matrix)

    print("TF-IDF built and normalized.", flush=True)

    return vectorizer, tfidf_matrix


# =====================================
# COMPUTE TOP-K SIMILARITIES (WITH SCORES)
# =====================================

def compute_topk_similarities(tfidf_matrix, top_k=TOP_K):

    print("Computing top-k similarities...", flush=True)

    n_movies = tfidf_matrix.shape[0]
    topk_dict = {}

    for i in range(n_movies):

        if i % 500 == 0:
            print(f"Processing {i}/{n_movies}", flush=True)

        # Cosine similarity (since normalized, dot product works)
        sim_scores = tfidf_matrix[i].dot(tfidf_matrix.T).toarray().flatten()

        # Get top-k + self
        top_indices = np.argpartition(
            sim_scores,
            -(top_k + 1)
        )[-(top_k + 1):]

        # Sort by similarity descending
        top_indices = top_indices[
            np.argsort(sim_scores[top_indices])[::-1]
        ]

        # Remove itself
        top_indices = top_indices[top_indices != i][:top_k]

        # Store index + similarity score
        topk_dict[i] = [
            (int(idx), float(sim_scores[idx]))
            for idx in top_indices
        ]

    print("Similarity computation completed.", flush=True)

    return topk_dict


# =====================================
# SAVE ARTIFACTS
# =====================================

def save_artifacts(vectorizer, tfidf_matrix, topk_dict, df):

    print("Saving artifacts...", flush=True)

    # Save vectorizer
    with open(os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    # Save normalized TF-IDF matrix
    with open(os.path.join(MODELS_PATH, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)

    # Save similarity dictionary
    with open(os.path.join(MODELS_PATH, "topk_similarities.pkl"), "wb") as f:
        pickle.dump(topk_dict, f)

    # Save metadata
    metadata = df[
        ["tconst", "title", "year", "rating", "votes"]
    ].copy()

    metadata.to_parquet(
        os.path.join(MODELS_PATH, "movie_metadata.parquet"),
        index=False
    )

    print("All artifacts saved successfully.", flush=True)


# =====================================
# MAIN
# =====================================

def main():

    df = load_features()

    vectorizer, tfidf_matrix = build_tfidf(df)

    topk_dict = compute_topk_similarities(tfidf_matrix)

    save_artifacts(vectorizer, tfidf_matrix, topk_dict, df)

    print("TRAINING COMPLETE", flush=True)


# =====================================

if __name__ == "__main__":
    main()