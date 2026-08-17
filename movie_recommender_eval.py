"""
Movie Recommender: Adding an Actual Evaluation
------------------------------------------------
The original notebook (movie-recommendation-project.ipynb) built a
content-based recommender using CountVectorizer + cosine similarity, but
only checked it by eyeballing 5 recommendations for 2 example movies.

This script adds a real, checkable evaluation: genre overlap between a
query movie and its top-5 recommendations, compared against a random
baseline and a popularity baseline, and compares two design choices
(genre+overview vs. overview-only tags; CountVectorizer vs. TF-IDF).

Note on data: the original notebook used a Kaggle-hosted dataset. This
script uses the public TMDB 5000 Movies dataset instead (same shape:
title, genres, overview) since the original Kaggle dataset wasn't
re-accessible while building this follow-up.

Author: Dostonbek URINOV
https://github.com/DostonUr | https://www.linkedin.com/in/doston-urinov/
"""

import ast
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "tmdb_5000_movies.csv"
N_SAMPLE_QUERIES = 30
N_RECOMMENDATIONS = 5
RANDOM_SEED = 42


def parse_genres(raw) -> list:
    try:
        return [item["name"] for item in ast.literal_eval(raw)]
    except Exception:
        return []


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["genre_list"] = df["genres"].apply(parse_genres)
    df = df[df["overview"].notna()].reset_index(drop=True)
    df["tags_overview_only"] = df["overview"].fillna("")
    df["tags_genre_overview"] = (
        df["genre_list"].apply(lambda g: " ".join(g)) + " " + df["overview"].fillna("")
    )
    return df


def genre_overlap(df: pd.DataFrame, query_idx: int, rec_idx: int) -> float:
    q = set(df.iloc[query_idx]["genre_list"])
    r = set(df.iloc[rec_idx]["genre_list"])
    if not q:
        return np.nan
    return len(q & r) / len(q)


def evaluate_content_based(df, tags_col, vectorizer_cls, sample_titles, label):
    vec = vectorizer_cls(max_features=10000, stop_words="english")
    X = vec.fit_transform(df[tags_col].values.astype("U")).toarray()
    sim = cosine_similarity(X)

    scores = []
    for title in sample_titles:
        idx = df[df["title"] == title].index
        if len(idx) == 0:
            continue
        idx = idx[0]
        ranked = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
        recs = [i for i, _ in ranked if i != idx][:N_RECOMMENDATIONS]
        scores += [genre_overlap(df, idx, r) for r in recs]

    scores = [s for s in scores if not np.isnan(s)]
    print(f"{label}: mean genre overlap = {np.mean(scores):.3f} (n={len(scores)})")
    return np.mean(scores)


def evaluate_random_baseline(df, sample_titles):
    random.seed(RANDOM_SEED)
    scores = []
    n = len(df)
    for title in sample_titles:
        idx = df[df["title"] == title].index
        if len(idx) == 0:
            continue
        idx = idx[0]
        recs = random.sample(range(n), N_RECOMMENDATIONS)
        scores += [genre_overlap(df, idx, r) for r in recs if r != idx]
    scores = [s for s in scores if not np.isnan(s)]
    print(f"Random baseline: mean genre overlap = {np.mean(scores):.3f} (n={len(scores)})")


def evaluate_popularity_baseline(df, sample_titles):
    top5 = df.sort_values("popularity", ascending=False).index[:N_RECOMMENDATIONS].tolist()
    scores = []
    for title in sample_titles:
        idx = df[df["title"] == title].index
        if len(idx) == 0:
            continue
        idx = idx[0]
        scores += [genre_overlap(df, idx, r) for r in top5 if r != idx]
    scores = [s for s in scores if not np.isnan(s)]
    print(f"Popularity baseline: mean genre overlap = {np.mean(scores):.3f} (n={len(scores)})")


def main():
    df = load_data()
    sample_titles = df["title"].sample(N_SAMPLE_QUERIES, random_state=RANDOM_SEED).tolist()

    print("=== Baselines ===")
    evaluate_random_baseline(df, sample_titles)
    evaluate_popularity_baseline(df, sample_titles)

    print("\n=== Content-based variants ===")
    evaluate_content_based(df, "tags_overview_only", CountVectorizer, sample_titles,
                            "CountVectorizer, overview only")
    evaluate_content_based(df, "tags_genre_overview", CountVectorizer, sample_titles,
                            "CountVectorizer, genre+overview")
    evaluate_content_based(df, "tags_genre_overview", TfidfVectorizer, sample_titles,
                            "TF-IDF, genre+overview")


if __name__ == "__main__":
    main()
