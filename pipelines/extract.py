import os
import gzip
import shutil
import logging
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")

os.makedirs(STAGING_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# Utility: Decompress .gz files
# --------------------------------------------------

def decompress_gz_files():
    logging.info("Decompressing .gz files...")

    for file in os.listdir(RAW_DIR):
        if file.endswith(".gz"):
            input_path = os.path.join(RAW_DIR, file)
            output_path = os.path.join(STAGING_DIR, file.replace(".gz", ""))

            if not os.path.exists(output_path):
                with gzip.open(input_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                logging.info(f"Decompressed: {file}")

    logging.info("Decompression completed.")


# --------------------------------------------------
# Step 1: Extract and Filter title.basics
# --------------------------------------------------

def extract_movies():
    logging.info("Extracting title.basics...")

    path = os.path.join(STAGING_DIR, "title.basics.tsv")

    cols = [
        "tconst",
        "primaryTitle",
        "titleType",
        "isAdult",
        "startYear",
        "runtimeMinutes",
        "genres"
    ]

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=cols,
        na_values="\\N",
        low_memory=False
    )

    # Early filtering
    df = df[
        (df["titleType"] == "movie") &
        (df["isAdult"] == 0) &
        (df["genres"].notna())
    ]

    df = df.drop(columns=["titleType", "isAdult"])

    df.to_parquet(os.path.join(STAGING_DIR, "movies_filtered.parquet"), index=False)

    logging.info(f"Movies extracted: {len(df)}")


# --------------------------------------------------
# Step 2: Extract ratings
# --------------------------------------------------

def extract_ratings():
    logging.info("Extracting ratings...")

    path = os.path.join(STAGING_DIR, "title.ratings.tsv")

    df = pd.read_csv(
        path,
        sep="\t",
        na_values="\\N"
    )

    df.to_parquet(os.path.join(STAGING_DIR, "ratings.parquet"), index=False)

    logging.info(f"Ratings extracted: {len(df)}")


# --------------------------------------------------
# Step 3: Extract crew
# --------------------------------------------------

def extract_crew():
    logging.info("Extracting crew...")

    path = os.path.join(STAGING_DIR, "title.crew.tsv")

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["tconst", "directors"],
        na_values="\\N"
    )

    df.to_parquet(os.path.join(STAGING_DIR, "crew.parquet"), index=False)

    logging.info(f"Crew extracted: {len(df)}")


# --------------------------------------------------
# Step 4: Extract principals (filtered)
# --------------------------------------------------

def extract_principals():
    logging.info("Extracting principals (actors only)...")

    path = os.path.join(STAGING_DIR, "title.principals.tsv")

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["tconst", "ordering", "nconst", "category"],
        na_values="\\N"
    )

    # Keep only actors
    df = df[df["category"].isin(["actor", "actress"])]

    df.to_parquet(os.path.join(STAGING_DIR, "principals_filtered.parquet"), index=False)

    logging.info(f"Principals extracted: {len(df)}")


# --------------------------------------------------
# Step 5: Extract names
# --------------------------------------------------

def extract_names():
    logging.info("Extracting names...")

    path = os.path.join(STAGING_DIR, "name.basics.tsv")

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["nconst", "primaryName"],
        na_values="\\N"
    )

    df.to_parquet(os.path.join(STAGING_DIR, "names.parquet"), index=False)

    logging.info(f"Names extracted: {len(df)}")


# --------------------------------------------------
# Main Runner
# --------------------------------------------------

if __name__ == "__main__":
    decompress_gz_files()
    extract_movies()
    extract_ratings()
    extract_crew()
    extract_principals()
    extract_names()

    logging.info("Extraction phase completed successfully.")
