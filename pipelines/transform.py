import os
import logging
import pandas as pd

# =====================================
# PATH CONFIG (PROJECT ROOT SAFE)
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

STAGING_PATH = os.path.join(PROJECT_ROOT, "data", "staging")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHUNK_SIZE = 500_000


# =====================================
# LOAD STAGING DATA (PARQUET ONLY)
# =====================================

def load_staging():

    logging.info("Loading staging parquet files...")

    movies = pd.read_parquet(f"{STAGING_PATH}/movies_filtered.parquet")
    ratings = pd.read_parquet(f"{STAGING_PATH}/ratings.parquet")
    crew = pd.read_parquet(f"{STAGING_PATH}/crew.parquet")
    names = pd.read_parquet(f"{STAGING_PATH}/names.parquet")

    return movies, ratings, crew, names


# =====================================
# MERGE RATINGS (FILTER EARLY)
# =====================================

def add_ratings(movies, ratings):

    logging.info("Filtering ratings (numVotes > 1000)...")

    ratings = ratings[ratings["numVotes"] > 1000]

    movies = movies.merge(ratings, on="tconst", how="inner")

    logging.info(f"Movies after rating filter: {len(movies)}")

    return movies


# =====================================
# ADD DIRECTOR (NORMALIZATION FIX)
# =====================================

def add_directors(movies, crew, names):

    logging.info("Adding directors...")

    crew = crew.dropna(subset=["directors"])

    # crew["directors"] = crew["directors"].str.split(",")
    crew.loc[:, "directors"] = crew["directors"].str.split(",")


    crew = crew.explode("directors")

    crew = crew.merge(
        names,
        left_on="directors",
        right_on="nconst",
        how="left"
    )

    crew = (
        crew.groupby("tconst")["primaryName"]
        .first()
        .reset_index()
        .rename(columns={"primaryName": "director"})
    )

    movies = movies.merge(crew, on="tconst", how="left")

    return movies


# =====================================
# ADD TOP 3 CAST (MEMORY SAFE CHUNKING)
# =====================================

def add_cast(movies, names):

    logging.info("Adding cast (chunk processing)...")

    movie_ids = set(movies["tconst"])

    cast_chunks = []

    principals_path = f"{STAGING_PATH}/principals_filtered.parquet"

    # Load parquet fully (already filtered in extraction)
    principals = pd.read_parquet(principals_path)

    principals = principals[principals["tconst"].isin(movie_ids)]

    principals.sort_values(["tconst", "ordering"], inplace=True)

    principals = principals.groupby("tconst").head(3)

    principals = principals.merge(names, on="nconst", how="left")

    cast = (
        principals.groupby("tconst")["primaryName"]
        .apply(lambda x: ", ".join(x.dropna()))
        .reset_index()
        .rename(columns={"primaryName": "cast"})
    )

    movies = movies.merge(cast, on="tconst", how="left")

    return movies


# =====================================
# FINALIZE DATASET
# =====================================

def finalize(movies):

    logging.info("Finalizing dataset...")

    final = movies[
        [
            "tconst",
            "primaryTitle",
            "startYear",
            "genres",
            "director",
            "cast",
            "averageRating",
            "numVotes"
        ]
    ].copy()

    final.rename(columns={
        "primaryTitle": "title",
        "startYear": "year",
        "averageRating": "rating",
        "numVotes": "votes"
    }, inplace=True)

    # Only fill text fields
    final["director"].fillna("Unknown", inplace=True)
    final["cast"].fillna("Unknown", inplace=True)

    logging.info(f"Final dataset shape: {final.shape}")

    return final


# =====================================
# SAVE
# =====================================

def save(final):

    output_path = f"{PROCESSED_PATH}/movies_final.parquet"

    final.to_parquet(output_path, index=False)

    logging.info(f"Saved final dataset to: {output_path}")


# =====================================
# MAIN
# =====================================

def main():

    movies, ratings, crew, names = load_staging()

    movies = add_ratings(movies, ratings)

    movies = add_directors(movies, crew, names)

    movies = add_cast(movies, names)

    final = finalize(movies)

    save(final)


if __name__ == "__main__":
    main()