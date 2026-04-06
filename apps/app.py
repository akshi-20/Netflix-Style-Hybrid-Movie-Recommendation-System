import os
import sys
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────
# PROJECT PATH SETUP
# ─────────────────────────────────────────

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.predict import (
    recommend_similar,
    popular_movies,
    trending_movies,
    recommend_by_genre,
    METADATA,
    FULL_DATA,
)

from apps.tmdb_utils import (
    load_tmdb_cache,
    fetch_tmdb_for_movie,
)

app = Flask(__name__)

# =====================================
# FALLBACK IMAGE
# =====================================

FALLBACK_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"

# =====================================
# LOAD DATA ONCE
# =====================================

print("🚀 Preparing movie data for app...")

ALL_MOVIES_DF = METADATA.copy()
ALL_MOVIES_DF["title_lower"] = ALL_MOVIES_DF["title"].astype(str).str.lower()

# Ensure tconst is string everywhere
ALL_MOVIES_DF["tconst"] = ALL_MOVIES_DF["tconst"].astype(str)
FULL_DATA["tconst"] = FULL_DATA["tconst"].astype(str)

# print("FULL_DATA columns:", FULL_DATA.columns.tolist())

# =====================================
# LOAD TMDb CACHE ONCE
# =====================================

TMDB_CACHE_DF = load_tmdb_cache()

if not TMDB_CACHE_DF.empty:
    TMDB_CACHE_DF["tconst"] = TMDB_CACHE_DF["tconst"].astype(str)

    # Merge TMDb fields into ALL_MOVIES_DF
    ALL_MOVIES_DF = ALL_MOVIES_DF.merge(
        TMDB_CACHE_DF,
        on="tconst",
        how="left"
    )

    # Merge TMDb fields into FULL_DATA too
    FULL_DATA = FULL_DATA.merge(
        TMDB_CACHE_DF,
        on="tconst",
        how="left"
    )

    print(f"🖼️ TMDb cache merged: {len(TMDB_CACHE_DF):,} posters available")
else:
    # Ensure columns exist even if cache is empty
    for col in [
        "tmdb_id",
        "poster_url",
        "backdrop_url",
        "overview",
        "tmdb_vote",
        "release_date",
    ]:
        if col not in ALL_MOVIES_DF.columns:
            ALL_MOVIES_DF[col] = None
        if col not in FULL_DATA.columns:
            FULL_DATA[col] = None

    print("⚠️ TMDb cache is empty. App will use fallback posters.")

print("✅ Movie data ready.")

# =====================================
# GENRE FILTER PILLS
# =====================================

GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy",
    "History", "Horror", "Music", "Mystery", "Romance",
    "Sci-Fi", "Sport", "Thriller", "War", "Western",
]

# =====================================
# ENRICH DATAFRAME WITH TMDb FIELDS
# =====================================

def enrich_with_tmdb(df: pd.DataFrame):
    """
    Merge poster/backdrop/overview from ALL_MOVIES_DF into
    dataframes returned by predict.py (like popular_movies/trending_movies).
    """
    if df is None or isinstance(df, str) or df.empty:
        return df

    df = df.copy()

    if "tconst" not in df.columns:
        return df

    tmdb_cols = [
        "tconst",
        "poster_url",
        "backdrop_url",
        "overview",
        "tmdb_vote",
        "release_date",
    ]

    # Ensure ALL_MOVIES_DF has these columns
    for col in tmdb_cols[1:]:
        if col not in ALL_MOVIES_DF.columns:
            ALL_MOVIES_DF[col] = None

    source = ALL_MOVIES_DF[tmdb_cols].copy()
    source["tconst"] = source["tconst"].astype(str).str.strip()
    df["tconst"] = df["tconst"].astype(str).str.strip()

    # Remove duplicate columns before merge
    for col in tmdb_cols[1:]:
        if col in df.columns:
            df = df.drop(columns=[col], errors="ignore")

    df = df.merge(source, on="tconst", how="left")

    return df

# =====================================
# DATAFRAME → LIST CONVERTER
# =====================================

def df_to_list(df: pd.DataFrame):
    if df is None or isinstance(df, str) or df.empty:
        return []

    movies = []

    for idx, row in df.iterrows():
        # Director
        director = ""
        if "director" in df.columns and pd.notna(row.get("director")):
            director = str(row.get("director")).strip()

        # Genres
        genres = ""
        if "genres" in df.columns and pd.notna(row.get("genres")):
            genres = str(row.get("genres")).replace("|", ", ").strip()

        # Poster
        poster_url = row.get("poster_url")
        poster_url = "" if pd.isna(poster_url) else str(poster_url).strip()

        # Reject bad / broken / malformed cached URLs
        if (
            not poster_url
            or poster_url.lower() in {"none", "nan", "null"}
            or poster_url.endswith("/None")
            or "image.tmdb.org" in poster_url and poster_url.count("/") < 5
        ):
            poster_url = FALLBACK_POSTER

        # Backdrop
        backdrop_url = row.get("backdrop_url")
        if pd.isna(backdrop_url) or not str(backdrop_url).strip():
            backdrop_url = ""

        # Overview
        overview = row.get("overview")
        if pd.isna(overview) or not str(overview).strip():
            overview = ""

        # TMDb vote
        tmdb_vote = row.get("tmdb_vote")
        if pd.isna(tmdb_vote):
            tmdb_vote = None

        # Release date
        release_date = row.get("release_date")
        if pd.isna(release_date) or not str(release_date).strip():
            release_date = ""

        movies.append({
            "id": str(row.get("tconst", idx)),
            "title": str(row.get("title", "Unknown")),
            "year": int(float(row.get("year", 0) or 0)),
            "rating": round(float(row.get("rating", 0) or 0), 1),
            "votes": int(row.get("votes", 0) or 0),
            "director": director,
            "genres": genres,

            # TMDb fields
            "poster_url": poster_url,
            "backdrop_url": backdrop_url,
            "overview": overview,
            "tmdb_vote": tmdb_vote,
            "release_date": release_date,
        })

    return movies

# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def index():
    # Data from predict.py
    top_rated_df = popular_movies(16)
    trending_df = trending_movies(16)

    # FIX: enrich METADATA rows with TMDb poster fields
    top_rated_df = enrich_with_tmdb(top_rated_df)
    trending_df = enrich_with_tmdb(trending_df)

    top_rated = df_to_list(top_rated_df)
    trending = df_to_list(trending_df)

    # DEBUG (optional) - remove later
    # print("\nTOP RATED POSTERS:")
    # for m in top_rated[:5]:
    #     print(m["title"], "=>", m["poster_url"])

    # print("\nTRENDING POSTERS:")
    # for m in trending[:5]:
    #     print(m["title"], "=>", m["poster_url"])

    if "genres" in FULL_DATA.columns:
        seen_genres = sorted({
            g.split("|")[0]
            for g in FULL_DATA["genres"].dropna().unique()
        })[:8]
    else:
        seen_genres = []

    by_genre = {}

    for genre in seen_genres:
        genre_df = recommend_by_genre(genre, 12)

        # safe for genre rows too
        genre_df = enrich_with_tmdb(genre_df)

        movies = df_to_list(genre_df)
        if movies:
            by_genre[genre] = movies

    return render_template(
        "index.html",
        top_rated=top_rated,
        trending=trending,
        by_genre=by_genre,
    )

# =====================================
# SEARCH / BROWSE PAGE
# =====================================

@app.route("/search")
def search():
    raw_query = request.args.get("q", "").strip()
    query = raw_query.lower()
    genre_filter = request.args.get("genre", "").strip()

    # CASE 1: Genre selected → use FULL_DATA directly
    if genre_filter:
        df = FULL_DATA.copy()

        # Apply genre filter
        df = df[df["genres"].astype(str).str.contains(genre_filter, case=False, na=False)]

        # Optional title search inside selected genre
        if query:
            df = df[df["title"].astype(str).str.lower().str.contains(query, na=False)]

        # Rank by quality score
        df["search_score"] = df["rating"] * np.log1p(df["votes"])
        df = df.sort_values("search_score", ascending=False).head(200)

    # CASE 2: No genre selected → use ALL_MOVIES_DF only to get matching IDs,
    # then fetch full rows from FULL_DATA so director is available
    else:
        meta_df = ALL_MOVIES_DF.copy()

        # Apply text search on metadata dataframe
        if query:
            meta_df = meta_df[meta_df["title_lower"].str.contains(query, na=False)]

        # Sort top movies
        if "pop_score" in meta_df.columns:
            meta_df = meta_df.sort_values("pop_score", ascending=False)
        elif "rating" in meta_df.columns:
            meta_df = meta_df.sort_values("rating", ascending=False)

        meta_df = meta_df.head(200)

        # Get selected movie IDs
        top_ids = meta_df["tconst"].astype(str).tolist()

        # Fetch matching rows from FULL_DATA
        df = FULL_DATA[FULL_DATA["tconst"].astype(str).isin(top_ids)].copy()

        # Preserve original order from meta_df
        order_map = {tid: i for i, tid in enumerate(top_ids)}
        df["order_rank"] = df["tconst"].astype(str).map(order_map)
        df = df.sort_values("order_rank").drop(columns=["order_rank"])

    # Ensure TMDb fields exist
    df = enrich_with_tmdb(df)

    results = df_to_list(df)

    return render_template(
        "search.html",
        results=results,
        query=raw_query,
        genre=genre_filter,
        genres=GENRES,
    )

# =====================================
# MOVIE DETAIL PAGE
# =====================================

@app.route("/movie/<path:movie_id>")
def movie_detail(movie_id):
    df = ALL_MOVIES_DF
    movie_row = df[df["tconst"] == movie_id]

    if movie_row.empty:
        return render_template("404.html"), 404

    # Convert current movie
    movie = df_to_list(movie_row)[0]

    # OPTIONAL LAZY FETCH:
    # If no real poster exists yet, fetch TMDb once for this movie only
    if movie["poster_url"] == FALLBACK_POSTER:
        raw_row = movie_row.iloc[0]
        tmdb_data = fetch_tmdb_for_movie(
            tconst=movie_id,
            title=raw_row.get("title"),
            year=raw_row.get("year"),
        )

        if tmdb_data:
            movie["poster_url"] = tmdb_data.get("poster_url") or FALLBACK_POSTER
            movie["backdrop_url"] = tmdb_data.get("backdrop_url") or ""
            movie["overview"] = tmdb_data.get("overview") or movie["overview"]
            movie["tmdb_vote"] = tmdb_data.get("tmdb_vote")
            movie["release_date"] = tmdb_data.get("release_date") or ""

    recommendations = recommend_similar(movie["title"], 12)

    if not isinstance(recommendations, str):
        # FIX: enrich recommendations too
        recommendations = enrich_with_tmdb(recommendations)
        recommendations = df_to_list(recommendations)
        recommendations = [
            r for r in recommendations
            if r["title"] != movie["title"]
        ]
    else:
        recommendations = []

    return render_template(
        "movie.html",
        movie=movie,
        recommendations=recommendations,
    )

# =====================================
# API ROUTES
# =====================================

@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip().lower()

    if not query or len(query) < 2:
        return jsonify([])

    df = ALL_MOVIES_DF
    df = df[df["title_lower"].str.contains(query, na=False)]
    df = df.sort_values("rating", ascending=False).head(8)

    return jsonify(df_to_list(df))

@app.route("/api/recommendations/<path:movie_id>")
def api_recommendations(movie_id):
    df = ALL_MOVIES_DF
    movie_row = df[df["tconst"] == movie_id]

    if movie_row.empty:
        return jsonify({"error": "Movie not found"}), 404

    title = movie_row.iloc[0]["title"]

    recs = recommend_similar(title, 12)

    if isinstance(recs, str):
        return jsonify([])

    recs = enrich_with_tmdb(recs)
    recs = df_to_list(recs)
    recs = [r for r in recs if r["title"] != title]

    return jsonify(recs)

@app.route("/api/popular")
def api_popular():
    df = popular_movies(20)
    df = enrich_with_tmdb(df)
    return jsonify(df_to_list(df))

@app.route("/api/trending")
def api_trending():
    df = trending_movies(20)
    df = enrich_with_tmdb(df)
    return jsonify(df_to_list(df))

# =====================================
# RUN SERVER
# =====================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)