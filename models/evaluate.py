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

TOP_N = 10


# =====================================
# LOAD DATA
# =====================================

def load_data():

    with open(os.path.join(MODELS_PATH, "topk_similarities.pkl"), "rb") as f:
        topk_dict = pickle.load(f)

    metadata = pd.read_parquet(
        os.path.join(MODELS_PATH, "movie_metadata.parquet")
    )

    full_data = pd.read_parquet(
        os.path.join(PROCESSED_PATH, "movies_final.parquet")
    )

    return topk_dict, metadata, full_data


# =====================================
# GENRE OVERLAP CHECK
# =====================================

def has_genre_overlap(genres_a, genres_b):

    set_a = set(str(genres_a).split(","))
    set_b = set(str(genres_b).split(","))

    return len(set_a.intersection(set_b)) > 0


# =====================================
# EVALUATE
# =====================================

def evaluate_model(sample_size=200):

    topk_dict, metadata, full_data = load_data()

    indices = np.random.choice(
        list(topk_dict.keys()),
        size=min(sample_size, len(topk_dict)),
        replace=False
    )

    genre_hits = 0
    director_hits = 0
    total_recommendations = 0
    similarity_scores = []

    for idx in indices:

        base_movie = full_data.iloc[idx]
        base_genres = base_movie["genres"]
        base_director = base_movie["director"]

        recommendations = topk_dict.get(idx, [])[:TOP_N]

        for rec_idx, sim_score in recommendations:

            rec_movie = full_data.iloc[rec_idx]

            # Genre overlap
            if has_genre_overlap(base_genres, rec_movie["genres"]):
                genre_hits += 1

            # Director match
            if base_director == rec_movie["director"]:
                director_hits += 1

            similarity_scores.append(sim_score)
            total_recommendations += 1

    # Metrics
    genre_precision = genre_hits / total_recommendations
    director_match_rate = director_hits / total_recommendations
    avg_similarity = np.mean(similarity_scores)

    print("\n===== EVALUATION RESULTS =====\n")
    print(f"Sampled Movies: {len(indices)}")
    print(f"Total Recommendations Evaluated: {total_recommendations}")
    print(f"Genre Precision@{TOP_N}: {genre_precision:.4f}")
    print(f"Director Match Rate: {director_match_rate:.4f}")
    print(f"Average Similarity Score: {avg_similarity:.4f}")
    print("\n================================\n")


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    evaluate_model()