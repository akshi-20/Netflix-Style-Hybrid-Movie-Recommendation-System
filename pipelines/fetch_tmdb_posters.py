import os
import sys
import time
import pandas as pd

# =====================================
# PATH SETUP
# =====================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Allow importing from apps/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from apps.tmdb_utils import fetch_tmdb_for_movie, load_tmdb_cache

# =====================================
# DATA PATHS
# =====================================

MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "movies_final.parquet")

# =====================================
# CONFIG
# =====================================

TOP_N = 1000          # preload top 1000 movies
REQUEST_DELAY = 0.20  # 200ms between calls (safe and polite)

# =====================================
# HELPERS
# =====================================

def load_movies():
    if not os.path.exists(MOVIES_PATH):
        raise FileNotFoundError(f"Movies dataset not found: {MOVIES_PATH}")

    df = pd.read_parquet(MOVIES_PATH)

    # Ensure required columns exist
    required_cols = ["tconst", "title"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in movies dataset: {col}")

    # Optional columns
    if "year" not in df.columns:
        df["year"] = None

    if "votes" not in df.columns:
        df["votes"] = 0

    # Clean
    df["tconst"] = df["tconst"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()

    return df


def get_top_movies(df, top_n=TOP_N):
    """
    Pick the most important movies to preload first.
    Priority: highest votes.
    """
    df = df.copy()

    # Make sure votes are numeric
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0)

    # Sort by votes descending
    top_df = df.sort_values(by="votes", ascending=False).head(top_n).copy()

    return top_df


def filter_uncached_movies(df):
    """
    Remove movies already present in tmdb_cache.
    """
    cache_df = load_tmdb_cache()

    if cache_df.empty:
        return df.copy()

    cached_ids = set(cache_df["tconst"].astype(str).str.strip())
    filtered = df[~df["tconst"].isin(cached_ids)].copy()

    return filtered


# =====================================
# MAIN SCRIPT
# =====================================

def main():
    print("=" * 60)
    print("TMDb Poster Preload Script Started")
    print("=" * 60)

    # 1. Load movies
    movies_df = load_movies()
    print(f"[INFO] Loaded movies dataset: {len(movies_df):,} rows")

    # 2. Pick top N
    top_movies_df = get_top_movies(movies_df, TOP_N)
    print(f"[INFO] Selected top {len(top_movies_df):,} movies by votes")

    # 3. Remove already cached
    pending_df = filter_uncached_movies(top_movies_df)
    print(f"[INFO] Movies still needing TMDb fetch: {len(pending_df):,}")

    if pending_df.empty:
        print("[INFO] All selected movies already cached. Nothing to do.")
        return

    # 4. Fetch TMDb one by one
    success_count = 0
    fail_count = 0

    total = len(pending_df)

    for idx, row in enumerate(pending_df.itertuples(index=False), start=1):
        tconst = str(getattr(row, "tconst", "")).strip()
        title = str(getattr(row, "title", "")).strip()
        year = getattr(row, "year", None)

        print("-" * 60)
        print(f"[{idx}/{total}] Fetching TMDb for {title} ({year}) [{tconst}]")

        result = fetch_tmdb_for_movie(
            tconst=tconst,
            title=title,
            year=year,
        )

        if result:
            success_count += 1
            print(f"[SUCCESS] Cached poster for: {title}")
        else:
            fail_count += 1
            print(f"[MISS] No TMDb match found for: {title}")

        # polite delay
        time.sleep(REQUEST_DELAY)

    # 5. Final summary
    print("=" * 60)
    print("TMDb Poster Preload Script Finished")
    print("=" * 60)
    print(f"[SUMMARY] Success: {success_count}")
    print(f"[SUMMARY] Failed : {fail_count}")
    print(f"[SUMMARY] Total  : {total}")


if __name__ == "__main__":
    main()