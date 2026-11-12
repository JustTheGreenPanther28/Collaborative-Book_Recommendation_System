"""
Streamlit frontend for the Collaborative-Filtering Book Recommendation System.
Repo: https://github.com/JustTheGreenPanther28/Collaborative-Book_Recommendation_System

Run:
    pip install streamlit pandas numpy scikit-learn
    streamlit run app.py

Place app.py in the same folder as Books.csv and Ratings.csv (the repo root).
The model is built once on first launch (same logic as the notebook) and
cached to ./artifacts so later launches are instant.
"""

import html
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "model.pkl"

MIN_VOTES_POPULAR = 250   # popularity: minimum number of ratings
TOP_N_POPULAR = 50        # popularity: how many books to show
MIN_USER_RATINGS = 200    # collaborative: "expert" users rated > 200 books
MIN_BOOK_RATINGS = 50     # collaborative: books with >= 50 ratings by those users

PLACEHOLDER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='160' height='240'>"
    "<rect width='100%' height='100%' fill='%23e5e7eb'/>"
    "<text x='50%' y='50%' font-size='14' text-anchor='middle' fill='%236b7280'>"
    "No cover</text></svg>"
)

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------
# Data + model
# --------------------------------------------------------------------------
def find_csv(name: str) -> Path:
    """Find a CSV regardless of letter case (Books.csv vs books.csv)."""
    for p in BASE_DIR.iterdir():
        if p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(
        f"Could not find {name} next to app.py. Put Books.csv and Ratings.csv "
        "in the same folder as this file."
    )


def build_model() -> dict:
    """Reproduces the notebook's popularity + collaborative filtering pipeline."""
    books = pd.read_csv(find_csv("Books.csv"), low_memory=False)
    ratings = pd.read_csv(find_csv("Ratings.csv"))

    ratings_with_name = ratings.merge(books, on="ISBN")

    # ---- Popularity based --------------------------------------------------
    num_rating_df = (
        ratings_with_name.groupby("Book-Title")["Book-Rating"]
        .count()
        .reset_index()
        .rename(columns={"Book-Rating": "num-rating"})
    )
    avg_rating_df = (
        ratings_with_name.groupby("Book-Title")["Book-Rating"]
        .mean()
        .reset_index()
        .rename(columns={"Book-Rating": "avg-rating"})
    )
    popular_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
    popular_df = (
        popular_df[popular_df["num-rating"] > MIN_VOTES_POPULAR]
        .sort_values("avg-rating", ascending=False)
        .head(TOP_N_POPULAR)
    )
    popular_df = popular_df.merge(books, on="Book-Title").drop_duplicates("Book-Title")[
        ["Book-Title", "Book-Author", "Image-URL-M", "num-rating", "avg-rating"]
    ]

    # ---- Collaborative filtering ------------------------------------------
    counts_per_user = ratings_with_name.groupby("User-ID")["Book-Rating"].count()
    bookish_users = counts_per_user[counts_per_user > MIN_USER_RATINGS].index
    filtered_users = ratings_with_name[ratings_with_name["User-ID"].isin(bookish_users)]

    counts_per_book = filtered_users.groupby("Book-Title")["Book-Rating"].count()
    famous_books = counts_per_book[counts_per_book >= MIN_BOOK_RATINGS].index
    final_ratings = filtered_users[filtered_users["Book-Title"].isin(famous_books)]

    pt = final_ratings.pivot_table(
        index="Book-Title", columns="User-ID", values="Book-Rating"
    ).fillna(0)
    similarity_scores = cosine_similarity(pt)

    # Lookup table: title -> author / cover (for rendering recommendations)
    book_info = (
        books.drop_duplicates("Book-Title")[["Book-Title", "Book-Author", "Image-URL-M"]]
        .set_index("Book-Title")
    )
    book_info = book_info.loc[book_info.index.intersection(pt.index)]

    return {
        "popular_df": popular_df.reset_index(drop=True),
        "titles": pt.index.tolist(),
        "similarity_scores": similarity_scores.astype(np.float32),
        "book_info": book_info,
    }


@st.cache_resource(show_spinner=False)
def load_model() -> dict:
    """Load cached model from disk, or build it (first run only)."""
    if ARTIFACT_FILE.exists():
        with open(ARTIFACT_FILE, "rb") as f:
            return pickle.load(f)

    model = build_model()
    try:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        with open(ARTIFACT_FILE, "wb") as f:
            pickle.dump(model, f)
    except OSError:
        pass  # read-only filesystem (e.g. some hosts) - just keep in memory
    return model


def recommend(book_name: str, model: dict, n: int = 5) -> list[dict]:
    """Same logic as the notebook's recommend(), but returns data instead of printing."""
    titles = model["titles"]
    idx = titles.index(book_name)
    scores = model["similarity_scores"][idx]
    similar = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1 : n + 1]

    results = []
    for i, score in similar:
        title = titles[i]
        info = model["book_info"].loc[title] if title in model["book_info"].index else None
        results.append(
            {
                "title": title,
                "author": info["Book-Author"] if info is not None else "Unknown",
                "image": info["Image-URL-M"] if info is not None else "",
                "score": float(score),
            }
        )
    return results


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def book_card(title: str, author: str, image: str, *lines: str) -> str:
    img = html.escape(str(image)) if isinstance(image, str) and image else PLACEHOLDER
    extra = "".join(f"<div class='meta'>{html.escape(l)}</div>" for l in lines)
    return f"""
    <div class="card">
        <img src="{img}" loading="lazy" referrerpolicy="no-referrer"
             onerror="this.onerror=null;this.src=&quot;{PLACEHOLDER}&quot;;" />
        <div class="title" title="{html.escape(title)}">{html.escape(title)}</div>
        <div class="author">{html.escape(str(author))}</div>
        {extra}
    </div>
    """


CSS = """
<style>
.card {text-align:center; padding:10px 6px 14px; border-radius:12px;
       border:1px solid rgba(128,128,128,.25); height:100%;}
.card img {width:130px; height:195px; object-fit:cover; border-radius:6px;
           box-shadow:0 2px 8px rgba(0,0,0,.25);}
.card .title {font-weight:600; margin-top:8px; font-size:.92rem; line-height:1.25;
              display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
              overflow:hidden; min-height:2.5em;}
.card .author {font-size:.8rem; opacity:.75; margin-top:2px;}
.card .meta {font-size:.78rem; margin-top:4px; opacity:.9;}
</style>
"""


def render_grid(cards: list[str], per_row: int = 5):
    for start in range(0, len(cards), per_row):
        cols = st.columns(per_row)
        for col, card in zip(cols, cards[start : start + per_row]):
            col.markdown(card, unsafe_allow_html=True)
        st.write("")


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
st.title("📚 Book Recommender System")
st.caption("Popularity-based and collaborative-filtering (cosine similarity) recommendations.")

try:
    with st.spinner("Loading model… (first launch builds it from the CSVs and may take a minute)"):
        model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab_recommend, tab_popular = st.tabs(["🔎 Recommend", "🔥 Top 50 Popular"])

# ---- Recommend tab ---------------------------------------------------------
with tab_recommend:
    st.subheader("Find books similar to one you like")

    left, right = st.columns([3, 1])
    with left:
        selected = st.selectbox(
            "Pick a book (start typing to search)",
            options=model["titles"],
            index=None,
            placeholder="Type a book title…",
        )
    with right:
        n_recs = st.slider("How many?", 3, 10, 5)

    if selected:
        info = model["book_info"].loc[selected] if selected in model["book_info"].index else None
        if info is not None:
            st.markdown("**You selected:**")
            c1, _ = st.columns([1, 4])
            c1.markdown(book_card(selected, info["Book-Author"], info["Image-URL-M"]),
                        unsafe_allow_html=True)

        st.markdown(f"### Because you liked *{selected}*")
        recs = recommend(selected, model, n_recs)
        cards = [
            book_card(r["title"], r["author"], r["image"], f"Similarity: {r['score']:.2f}")
            for r in recs
        ]
        render_grid(cards, per_row=5)
    else:
        st.info("Select a book above to get recommendations.")

    st.caption(
        f"{len(model['titles'])} books available for recommendations "
        f"(rated by users with > {MIN_USER_RATINGS} ratings, books with ≥ {MIN_BOOK_RATINGS} ratings)."
    )

# ---- Popular tab -----------------------------------------------------------
with tab_popular:
    st.subheader(f"Top {TOP_N_POPULAR} books by average rating")
    st.caption(f"Only books with more than {MIN_VOTES_POPULAR} ratings are considered.")

    cards = [
        book_card(
            row["Book-Title"],
            row["Book-Author"],
            row["Image-URL-M"],
            f"⭐ {row['avg-rating']:.2f}",
            f"{int(row['num-rating'])} ratings",
        )
        for _, row in model["popular_df"].iterrows()
    ]
    render_grid(cards, per_row=5)
