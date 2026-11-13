# 📚 Collaborative Book Recommendation System

A book recommender built with **collaborative filtering** (item-based, cosine similarity) and a **popularity-based** ranking, served through a [Streamlit](https://streamlit.io/) web app.

🔗 **Live demo:** https://collaborative-bookrecommendationsystem.streamlit.app/

---

## Features

- **🔎 Recommend:** pick a book and get the most similar titles, with cover, author and similarity score. Choose how many results to show (3–10).
- **🔥 Top 50 Popular:** the highest-rated books among those with a large number of ratings.
- **⚡ Cached model:** the model is built once from the CSVs and saved to `artifacts/model.pkl`, so later launches are instant.

## How it works

### 1. Popularity-based recommender
1. Merge `Ratings.csv` with `Books.csv` on `ISBN`.
2. Compute the number of ratings and the average rating per book title.
3. Keep books with **more than 250 ratings**.
4. Sort by average rating and show the **top 50**.

### 2. Collaborative filtering recommender (item-based)
1. Keep only "experienced" readers: users who rated **more than 200 books**.
2. Keep only books with **at least 50 ratings** from those users.
3. Build a pivot table of **book × user** ratings (missing values filled with `0`).
4. Compute **cosine similarity** between every pair of books.
5. For a selected book, return the `n` books with the highest similarity scores (excluding the book itself).

The thresholds are set as constants at the top of `app.py`:

| Constant | Value | Meaning |
| --- | --- | --- |
| `MIN_VOTES_POPULAR` | 250 | Minimum ratings for the popularity list |
| `TOP_N_POPULAR` | 50 | Books shown in the popularity tab |
| `MIN_USER_RATINGS` | 200 | A user must have rated more than this many books |
| `MIN_BOOK_RATINGS` | 50 | A book needs at least this many ratings from those users |

## Project structure

```
├── app.py                 # Streamlit frontend + model building/loading
├── Collaborative Filtering Based Book recommendation system.ipynb
│                          # Notebook with the original analysis and pipeline
├── Books.csv              # Book metadata (title, author, cover URL, ...)
├── Ratings.csv            # User ratings (User-ID, ISBN, Book-Rating)
├── Users.csv              # User data
├── artifacts/             # Cached model (model.pkl), created on first run
└── requirements.txt       # Python dependencies
```

Columns used by the app: `ISBN`, `Book-Title`, `Book-Author`, `Image-URL-M` (from `Books.csv`) and `User-ID`, `ISBN`, `Book-Rating` (from `Ratings.csv`).

## Getting started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/JustTheGreenPanther28/Collaborative-Book_Recommendation_System.git
cd Collaborative-Book_Recommendation_System
pip install -r requirements.txt
```

Or install the dependencies directly:

```bash
pip install streamlit pandas numpy scikit-learn
```

### Run the app

```bash
streamlit run app.py
```

Make sure `Books.csv` and `Ratings.csv` are in the same folder as `app.py`. The first launch builds the model from the CSVs and may take a minute; it is then cached to `artifacts/model.pkl`.

> If you change the data or the threshold constants, delete `artifacts/model.pkl` so the model is rebuilt.

## Tech stack

- **Python**
- **pandas / NumPy:** data processing
- **scikit-learn:** cosine similarity
- **Streamlit:** web interface

## Limitations and ideas for improvement

- Missing ratings are filled with `0`, so "not rated" is treated like "rated zero". Mean-centering or sparse similarity would be more accurate.
- Books are matched by **title**, so different editions of the same book are merged.
- Only books with enough ratings from experienced users can be recommended (cold-start items are not covered).
- The popularity list ranks by raw average rating; a weighted score (e.g. Bayesian average) would be more robust.
- Possible extensions: user-based filtering, matrix factorization (SVD/ALS), hybrid content + collaborative approaches, and offline evaluation metrics (precision@k, recall@k).

## Contributing

Issues and pull requests are welcome. Feel free to fork the repo and open a PR.

## License

No license has been specified yet. Consider adding one (e.g. MIT) if you want others to reuse the code.
