# Movie Recommendation System — Now With an Actual Evaluation

A content-based movie recommender (genre + plot overview → CountVectorizer → cosine similarity),
originally evaluated only by eyeballing 5 recommendations for 2 example movies. This follow-up
adds a real, checkable evaluation against baselines, and tests two design choices that weren't
originally questioned.

Full write-up: **[Read the article](https://urinovd.github.io/2026/12/05/movie-recommender-actual-evaluation.html)**

## Data note

The original notebook (`movie-recommendation-project.ipynb`) used a Kaggle-hosted dataset. The
evaluation script (`movie_recommender_eval.py`) uses the public **TMDB 5000 Movies** dataset
instead — same shape (title, genres, overview) — since the original dataset wasn't re-accessible
while building this follow-up. Findings below are from the TMDB 5000 dataset.

## What's here

- `movie-recommendation-project.ipynb` — the original recommender notebook
- `movie_recommender_eval.py` — the evaluation: genre-overlap scoring against random and
  popularity baselines, plus a comparison of tag composition and vectorizer choice
- `tmdb_5000_movies.csv` — dataset used for the evaluation

## Method

For 30 randomly sampled query movies, each approach's top-5 recommendations are scored by
**genre overlap**: the fraction of the query movie's genres present in each recommendation's
genres. Compared against:
- **Random baseline** — 5 randomly chosen movies
- **Popularity baseline** — always the 5 most popular movies, regardless of query

## Results

| Approach | Mean genre overlap (top-5) |
|---|---|
| Random baseline | 30.0% |
| Popularity baseline | 26.1% |
| Overview text only (CountVectorizer) | 42.7% |
| **Genre + overview (CountVectorizer)** | **76.1%** |
| Genre + overview (TF-IDF) | 49.5% |

Two findings worth noting:
1. The content-based approach clearly beats both baselines — real evidence the recommender adds
   value, which the original version never measured.
2. **TF-IDF performed worse than plain word counts** — likely because TF-IDF downweights the
   short, frequently-repeated genre terms that this system needs weighted heavily, in favor of
   rarer overview vocabulary. The "usually better" default wasn't better here.

## Run it

```bash
pip install -r requirements.txt
python movie_recommender_eval.py
```

## Author

Dostonbek URINOV — [LinkedIn](https://www.linkedin.com/in/doston-urinov/) ·
[GitHub](https://github.com/DostonUr) · [Kaggle](https://www.kaggle.com/dostonur)
