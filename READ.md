# 🎬 Netflix-Style Hybrid Movie Recommendation System

> An end-to-end data engineering and content-based recommendation pipeline built on IMDb public datasets, enriched with TMDb metadata, and served via a lightweight Flask demo interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Parquet-green?style=flat-square&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-TF--IDF-orange?style=flat-square&logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-Demo%20Layer-lightgrey?style=flat-square&logo=flask)
![IMDb](https://img.shields.io/badge/Data-IMDb%20Datasets-yellow?style=flat-square)
![TMDb](https://img.shields.io/badge/API-TMDb-teal?style=flat-square)

---

## 📌 Project Overview

This project demonstrates a **production-style data pipeline and machine learning system** for movie recommendations. The core focus is on:

- Robust **multi-stage ETL pipeline** ingesting and transforming large-scale IMDb data
- **Feature engineering** for NLP-based content similarity modeling
- **Offline precomputation** of recommendations for instant, scalable serving
- **External API integration** with caching to enrich the dataset with TMDb metadata

The Flask web app (CineWatch) is a **thin demo layer** — the engineering and data science work is what drives it.

---

## 🗂️ Project Structure

```
MOVIE_RECOMMENDER/
│
├── apps/                          # Flask demo layer (serving only)
│   ├── app.py                     # Routes + rendering
│   ├── tmdb_utils.py              # TMDb API fetch + cache helpers
│   └── templates/                 # Jinja2 HTML templates
│
├── data/
│   ├── raw/                       # Raw IMDb TSV source files
│   ├── staging/                   # Intermediate pipeline outputs
│   └── processed/
│       ├── movies_final.parquet   # Full processed movie dataset
│       ├── movies_clean.parquet   # Cleaned, normalized dataset
│       ├── movies_features.parquet # Engineered feature set
│       └── tmdb_cache.parquet     # Cached TMDb API enrichment data
│
├── models/
│   ├── train.py                   # Model training pipeline
│   ├── predict.py                 # Recommendation inference
│   ├── evaluate.py                # Evaluation logic
│   ├── movie_metadata.parquet     # Metadata for ranking
│   ├── tfidf_matrix.pkl           # TF-IDF sparse matrix
│   ├── tfidf_vectorizer.pkl       # Fitted vectorizer artifact
│   └── topk_similarities.pkl     # Precomputed top-k similar movies
│
├── pipelines/
│   ├── extract.py                 # Raw data extraction
│   ├── transform.py               # Cleaning and normalization
│   ├── build_features.py          # Feature engineering
│   └── fetch_tmdb_posters.py      # TMDb API enrichment script
│
├── docs/                          # Notes, reports, screenshots
├── .env                           # TMDb API secrets (gitignored)
├── .gitignore
└── requirements.txt
```

---

## 🏗️ Data Engineering Pipeline

The pipeline follows a structured **raw → staging → processed** zone architecture, mirroring modern data lakehouse design.

```
IMDb TSVs (raw)
     │
     ▼
extract.py          ← Multi-file ingestion (title.basics, title.ratings, etc.)
     │
     ▼
transform.py        ← Null handling, type normalization, join logic
     │
     ▼
build_features.py   ← Text feature construction, genre encoding, metadata assembly
     │
     ▼
movies_features.parquet  ← Clean, model-ready dataset
     │
     ▼
fetch_tmdb_posters.py    ← TMDb API enrichment → tmdb_cache.parquet
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Parquet over CSV** | Columnar format — faster reads, lower memory, native type preservation |
| **Staged zones (raw / staging / processed)** | Reproducibility; enables reprocessing from any checkpoint |
| **TMDb Parquet cache** | Avoids redundant API calls, handles rate limits, persists across pipeline runs |
| **Modular pipeline scripts** | Each stage independently testable and rerunnable |

---

## 🔬 Data Science & Modeling

### Feature Engineering

Text features are constructed from structured IMDb fields — genres, titles, and contextual metadata — and assembled into a unified representation in `build_features.py`, producing `movies_features.parquet`.

### TF-IDF Vectorization

```python
# Fitted on the full processed corpus
TfidfVectorizer → tfidf_matrix.pkl (sparse matrix)
               → tfidf_vectorizer.pkl (fitted transformer)
```

- Converts high-dimensional text features into a sparse TF-IDF matrix
- Captures content similarity across the entire IMDb corpus

### Cosine Similarity + Precomputed Top-K

```python
# Offline computation → persisted artifact
cosine_similarity(tfidf_matrix) → topk_similarities.pkl
```

- Pairwise cosine similarity computed across all movies
- **Top-K results are precomputed and serialized offline** — inference at serving time is a pure dictionary lookup, not a matrix computation
- This decouples expensive computation from request latency

### Offline vs. Online Inference Pattern

```
OFFLINE (train.py)                    ONLINE (predict.py + Flask)
─────────────────                     ──────────────────────────
Fit TF-IDF vectorizer          →      Load tfidf_vectorizer.pkl
Compute similarity matrix      →      Load topk_similarities.pkl
Persist top-K per movie        →      Return precomputed results instantly
```

---

## 📊 Demo: CineWatch

The Flask app serves as a **proof-of-concept interface** to demonstrate recommendation quality.

**Movie Detail Page** — IMDb rating, vote count, release year, TMDb poster
**"Because You Watched..." Section** — Top-K content-based recommendations rendered as a Netflix-style card grid

> *Screenshot: Recommendations for "Inception" → Interstellar, The Dark Knight, The Prestige*
> *Screenshot: Recommendations for "Rockstar" → Jab We Met, Highway, Tamasha*

The app highlights that the recommendation engine works meaningfully across both **Hollywood** and **Bollywood** content — a direct result of IMDb's multilingual dataset coverage.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, PyArrow (Parquet) |
| ML / NLP | Scikit-learn (TF-IDF, cosine similarity) |
| Data Source | [IMDb Datasets](https://datasets.imdbws.com/) |
| API Enrichment | [TMDb API](https://www.themoviedb.org/documentation/api) |
| Serving Layer | Flask, Jinja2 |
| Storage Formats | Parquet (data), Pickle (model artifacts) |
| Config | python-dotenv (`.env` for secrets) |

---

## ⚙️ Setup & Reproduction

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
# .env
TMDB_BEARER_TOKEN=your_token_here
```

### 3. Run the ETL pipeline
```bash
python pipelines/extract.py
python pipelines/transform.py
python pipelines/build_features.py
python pipelines/fetch_tmdb_posters.py
```

### 4. Train the model
```bash
python models/train.py
```

### 5. (Optional) Evaluate
```bash
python models/evaluate.py
```

### 6. Launch the demo app
```bash
python apps/app.py
# Navigate to http://127.0.0.1:5000
```

---

## 📁 Data Sources

- **IMDb Non-Commercial Datasets** — `https://datasets.imdbws.com/`
  Files used: `title.basics.tsv.gz`, `title.ratings.tsv.gz`, and related title files
- **TMDb API** — Poster images and movie overviews fetched via bearer token auth and cached locally

> IMDb data is used under their [Non-Commercial Licensing Terms](https://www.imdb.com/interfaces/).

---

## 🔭 Future Scope

- [ ] Incorporate collaborative filtering signals (user ratings) for a hybrid model
- [ ] Replace pickle artifacts with a lightweight vector database (e.g., FAISS, ChromaDB)
- [ ] Add automated pipeline orchestration (Airflow / Prefect)
- [ ] Expand evaluation metrics (precision@K, MAP, NDCG)
- [ ] Containerize with Docker for reproducible deployment

---

## 👤 Author

Built as a data engineering and data science portfolio project.
Focused on pipeline design, NLP-based feature engineering, and scalable offline inference patterns.