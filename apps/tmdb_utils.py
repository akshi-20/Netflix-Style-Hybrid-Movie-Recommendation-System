import os
import time
import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

# =====================================
# PATH CONFIG
# =====================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tmdb_cache.parquet")

# =====================================
# TMDb CONFIG (API KEY v3 AUTH)
# =====================================

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
print("[TMDB] API key loaded:", "YES" if TMDB_API_KEY else "NO")


BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500"
BASE_BACKDROP_URL = "https://image.tmdb.org/t/p/original"

# =====================================
# CACHE CONFIG
# =====================================

CACHE_COLUMNS = [
    "tconst",
    "tmdb_id",
    "poster_url",
    "backdrop_url",
    "overview",
    "tmdb_vote",
    "release_date",
]

# =====================================
# CACHE HELPERS
# =====================================

def load_tmdb_cache():
    if os.path.exists(CACHE_PATH):
        try:
            df = pd.read_parquet(CACHE_PATH)
            for col in CACHE_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            print(f"[TMDB CACHE] Loaded {len(df)} cached rows")
            return df[CACHE_COLUMNS].copy()
        except Exception as e:
            print(f"[TMDB CACHE] Failed to load cache: {e}")

    return pd.DataFrame(columns=CACHE_COLUMNS)


def save_tmdb_cache(df):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)


def get_tmdb_from_cache(tconst):
    tconst = str(tconst).strip()
    cache_df = load_tmdb_cache()

    if cache_df.empty:
        return None

    match = cache_df[cache_df["tconst"].astype(str) == tconst]

    if not match.empty:
        print(f"[TMDB CACHE] HIT for {tconst}")
        return match.iloc[0].to_dict()

    print(f"[TMDB CACHE] MISS for {tconst}")
    return None


def append_to_tmdb_cache(row_dict):
    tconst = str(row_dict.get("tconst", "")).strip()
    if not tconst:
        return

    cache_df = load_tmdb_cache()

    # remove old row if exists
    if not cache_df.empty:
        cache_df = cache_df[cache_df["tconst"].astype(str) != tconst]

    new_row = pd.DataFrame([row_dict], columns=CACHE_COLUMNS)

    if cache_df.empty:
        cache_df = new_row.copy()
    else:
        cache_df = pd.concat([cache_df, new_row], ignore_index=True)

    save_tmdb_cache(cache_df)
    print(f"[TMDB CACHE] Saved {tconst} to cache")

# =====================================
# TMDb HELPERS
# =====================================

def build_poster_url(path):
    if path:
        return f"{BASE_POSTER_URL}{path}"
    return ""


def build_backdrop_url(path):
    if path:
        return f"{BASE_BACKDROP_URL}{path}"
    return ""


def normalize_tmdb_movie(movie, tconst):
    return {
        "tconst": tconst,
        "tmdb_id": movie.get("id"),
        "poster_url": build_poster_url(movie.get("poster_path")),
        "backdrop_url": build_backdrop_url(movie.get("backdrop_path")),
        "overview": movie.get("overview", ""),
        "tmdb_vote": movie.get("vote_average", ""),
        "release_date": movie.get("release_date", ""),
    }

# =====================================
# TMDb API CHECK
# =====================================

def ensure_api_key():
    if not TMDB_API_KEY:
        raise ValueError(
            "TMDB_API_KEY is missing. Set it as an environment variable."
        )

# =====================================
# TMDb FETCH HELPERS
# =====================================

def fetch_tmdb_by_imdb(tconst):
    ensure_api_key()

    url = f"https://api.themoviedb.org/3/find/{tconst}"
    params = {
        "api_key": TMDB_API_KEY,
        "external_source": "imdb_id"
    }

    try:
        res = requests.get(url, params=params, timeout=8)

        if res.status_code == 200:
            data = res.json()
            results = data.get("movie_results", [])
            if results:
                print(f"[TMDB] IMDb lookup success for {tconst}")
                return normalize_tmdb_movie(results[0], tconst)
        else:
            print(f"[TMDB] IMDb lookup failed {tconst}: {res.status_code} | {res.text}")

    except Exception as e:
        print(f"[TMDB] IMDb lookup error {tconst}: {e}")

    return None


def search_tmdb_by_title_year(title, year, tconst):
    ensure_api_key()

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
    }

    if pd.notna(year) and str(year).replace(".0", "").isdigit():
        params["year"] = int(float(year))

    try:
        res = requests.get(url, params=params, timeout=8)

        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])
            if results:
                print(f"[TMDB] Fallback search success for {title}")
                return normalize_tmdb_movie(results[0], tconst)
        else:
            print(f"[TMDB] Fallback search failed {title}: {res.status_code} | {res.text}")

    except Exception as e:
        print(f"[TMDB] Fallback search error {title}: {e}")

    return None

# =====================================
# MAIN FUNCTION
# =====================================

def fetch_tmdb_for_movie(tconst, title=None, year=None, delay=0.0):
    """
    Flow:
    1. Check cache
    2. Try IMDb ID lookup
    3. Fallback to title + year
    4. Save result to cache
    """
    tconst = str(tconst).strip()

    # 1. Cache first
    cached = get_tmdb_from_cache(tconst)
    if cached:
        return cached

    # 2. IMDb lookup
    result = fetch_tmdb_by_imdb(tconst)

    # 3. Fallback search
    if not result and title:
        result = search_tmdb_by_title_year(title, year, tconst)

    # Optional delay
    if delay > 0:
        time.sleep(delay)

    # 4. Save
    if result:
        append_to_tmdb_cache(result)

    return result