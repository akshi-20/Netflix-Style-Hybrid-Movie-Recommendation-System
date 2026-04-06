import os
import logging
import pandas as pd


# =====================================
# PATH CONFIG
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =====================================
# LOAD DATA
# =====================================

def load_data():
    path = os.path.join(PROCESSED_PATH, "movies_final.parquet")
    logging.info("Loading processed dataset...")
    df = pd.read_parquet(path)
    logging.info(f"Dataset shape: {df.shape}")
    return df


# =====================================
# CLEAN TEXT
# =====================================

def clean_text(df):

    logging.info("Cleaning text fields...")

    text_cols = ["genres", "director", "cast"]

    for col in text_cols:
        df[col] = df[col].fillna("").str.lower()

    df["genres"] = df["genres"].str.replace(",", " ", regex=False)
    df["cast"] = df["cast"].str.replace(",", " ", regex=False)

    df["director"] = df["director"].str.replace(" ", "", regex=False)
    df["cast"] = df["cast"].str.replace(" ", "", regex=False)

    return df


# =====================================
# CREATE WEIGHTED TAGS
# =====================================

def create_tags(df):

    logging.info("Creating weighted tags...")

    # Weighting strategy:
    # Director x2
    # Cast x2
    # Genres x1

    df["year_bucket"] = (df["year"] // 10 * 10).astype(str) + "s"

    df["tags"] = (
       df["genres"] + " " +
       df["director"] + " " + df["director"] + " " +
       df["cast"] + " " + df["cast"] + " " +
       df["year_bucket"]
    )

    return df


# =====================================
# SELECT COLUMNS
# =====================================

def select_columns(df):

    final = df[
        [
            "tconst",
            "title",
            "year",
            "rating",
            "votes",
            "tags"
        ]
    ].copy()

    logging.info(f"Feature dataset shape: {final.shape}")

    return final


# =====================================
# SAVE
# =====================================

def save(final):

    output_path = os.path.join(PROCESSED_PATH, "movies_features.parquet")
    final.to_parquet(output_path, index=False)
    logging.info(f"Saved features to: {output_path}")


# =====================================
# MAIN
# =====================================

def main():

    df = load_data()
    df = clean_text(df)
    df = create_tags(df)
    final = select_columns(df)
    save(final)


if __name__ == "__main__":
    main()