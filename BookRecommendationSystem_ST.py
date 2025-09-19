import streamlit as st
import joblib
import pandas as pd
from rapidfuzz import process, fuzz
import math
import random

st.set_page_config(
    page_title="Book Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    df_books = joblib.load('df_books.joblib')    
    df_booktags = joblib.load('df_booktags.joblib')    
    df_tags = joblib.load('df_tags.joblib')    
    cbf_matrix = joblib.load('cbf_matrix.joblib')
    cf_matrix = joblib.load('cf_matrix.joblib')
    return df_books, df_booktags,df_tags,cbf_matrix, cf_matrix

df_books,df_booktags,df_tags, cbf_matrix, cf_matrix = load_data()

# ----------------- System Functions -----------------
def image_viewer(booktitle: str):
    image_getter = df_books[df_books['original_title'] == booktitle]['image_url'].to_list()
    return image_getter[0] if image_getter else None

def get_hybrid_recommendations(book_title: str, author_query: str = None, n_recommendations=5,
                               cbf_matrix=cbf_matrix, cf_matrix=cf_matrix, dfBooks=df_books, weight=0.7):
    book_title = book_title.strip()
    book_titles = dfBooks['original_title'].tolist()
    author_titles = dfBooks['authors'].tolist()
    result = process.extractOne(book_title, book_titles, scorer=fuzz.WRatio)
    if result is None or result[1] < 70:
        if author_query is None:
            return None
        author_query = author_query.strip()
        secondresult = process.extractOne(author_query, author_titles, scorer=fuzz.WRatio)
        if secondresult is None or secondresult[1] < 70:
            return None
        idx = secondresult[2]
    else:
        idx = result[2]
    hybrid_sim_matrix = (weight * cbf_matrix) + ((1 - weight) * cf_matrix)
    sim_scores = sorted(list(enumerate(hybrid_sim_matrix[idx])), key=lambda x: x[1], reverse=True)[0:n_recommendations + 1]
    book_indices = [i[0] for i in sim_scores]
    return dfBooks.iloc[book_indices][['original_title','authors','original_publication_year','average_rating']].reset_index(drop=True)

def Top_Books_Rated_year(year: int):
    return df_books[df_books['original_publication_year'] == year]\
        .sort_values(by='average_rating', ascending=False)\
        .head()[['original_title','authors','average_rating']].reset_index(drop=True)

def Book_tags_getter(booktitle:str,Noftags=10):

    goodreads_book_id_getter=df_books[df_books['original_title']==booktitle]['goodreads_book_id'].to_list()[0]
    Booktags=pd.merge(df_booktags[df_booktags['goodreads_book_id']==goodreads_book_id_getter]['tag_id'],df_tags,how='inner',on='tag_id')['tag_name'][2:Noftags].tolist()

    return Booktags


# ----------------- UI,UX Functions -----------------
def render_colorful_tags(tags):
    if not tags:
        st.write("No tags found.")
        return

    colors = ["#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9",
              "#92A8D1", "#955251", "#B565A7", "#009B77",
              "#DD4124", "#45B8AC", "#EFC050", "#5B5EA6"]

    # build all badges first
    badges = ""
    for t in tags:
        color = random.choice(colors)
        badges += (
            f'<span style="background-color:{color};'
            'color:white;padding:4px 10px;margin:4px 6px;'
            'border-radius:12px;font-size:0.9rem;white-space:nowrap;">'
            f'{t}</span>'
        )

    # wrap them in a flex container
    html = (
        '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;">'
        f'{badges}'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# ----------------- Sidebar -----------------
st.sidebar.title("⚙️ Settings")
n_recs = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
weight = st.sidebar.slider("CBF Weight (vs CF)", 0.0, 1.0, 0.7, 0.1)

# Author input field
author_input = st.sidebar.text_input("Optional: Author Name", key="author_input")
find_by_author = st.sidebar.button("🔍 Find by Author Name", key="find_by_author_btn")


# Additional features/ideas
st.sidebar.markdown("---")
st.sidebar.write("**Contact:** [mohamedachrvf@gmail.com](mailto:mohamedachrvf@gmail.com)")
st.sidebar.write("**STAR MY REPO: ⭐🤩:**[GitHub](https://github.com/mohamedashraf2005/book-RecommendationSystem)")
st.sidebar.write("")

# ----------------- Main UI -----------------
st.title("Book Recommendation System📖")

st.markdown("##### Enter a Book title to get AI-personalized recommendations")

# Initialize session state for auto-trigger
if "auto_trigger_recs" not in st.session_state:
    st.session_state.auto_trigger_recs = False

text_input = st.text_input("", placeholder=" 🔎 Type Book Title and press Enter...", key="book_input")
selected_title = None

# Track if we should trigger author search
trigger_author_search = False

if text_input:
    all_titles = df_books['original_title'].tolist()
    suggestions = process.extract(text_input, all_titles, scorer=fuzz.WRatio, limit=10)
    suggestion_titles = [m[0] for m in suggestions if m[1] >= 70]

    if suggestion_titles:
        selected_title = st.selectbox("Suggested Books", suggestion_titles, key="suggestion_select")
        # Auto-trigger when user presses Enter after typing (text_input changes + suggestions exist)
        st.session_state.auto_trigger_recs = True
    else:
        st.warning("No matching books found. Try another keyword or add an author.")
        trigger_author_search = True

# Handle "Find by Author Name" button
if find_by_author:
    if not author_input.strip():
        st.error("Please enter an author name in the sidebar to search by author.")
    else:
        author_titles = df_books['authors'].tolist()
        author_match = process.extractOne(author_input.strip(), author_titles, scorer=fuzz.WRatio)
        if author_match and author_match[1] >= 70:
            selected_title = df_books.iloc[author_match[2]]['original_title']
            st.info(f"✅ Found book by author: **{selected_title}**")
            st.session_state.auto_trigger_recs = True  # Auto-trigger recommendations
        else:
            st.error("No matching author found. Try a different spelling or name.")

# ---- Buttons side-by-side ----
col1, col2 = st.columns([1,1])
with col1:
    view_recs = st.button("View Recommended Books", key="view_recs_btn")
with col2:
    surprise = st.button("Surprise me", key="surprise-btn")

# Auto-trigger logic: if flag is set and we have a selected_title, simulate button click
if st.session_state.auto_trigger_recs and selected_title:
    view_recs = True
    st.session_state.auto_trigger_recs = False  # Reset after use

# ---- View Recommendations ----
def render_books(recs_df, show_year=True):
    cols_per_row = 5
    rows = math.ceil(len(recs_df) / cols_per_row)
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c_idx, c in enumerate(cols):
            idx = r * cols_per_row + c_idx
            if idx < len(recs_df):
                book = recs_df.iloc[idx]
                title = book['original_title']
                authors = book['authors']
                rating = book['average_rating']
                image_url = image_viewer(title)
                with c:
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    st.image(image_url or "https://via.placeholder.com/150", width=150)
                    st.markdown(f'<div class="book-title">{title}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="book-author">{authors}</div>', unsafe_allow_html=True)
                    if show_year:
                        st.markdown(f'<div class="book-year">Year: {book["original_publication_year"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="book-rating">Rating: {rating:.2f} ⭐</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# Show recommendations if button clicked OR auto-triggered
if (view_recs or st.session_state.auto_trigger_recs) and selected_title:
    with st.spinner("Fetching recommendations..."):
        recs_df = get_hybrid_recommendations(selected_title, author_query=author_input, n_recommendations=n_recs, weight=weight)
    if recs_df is not None:
        st.success(f"Recommendations based on **{selected_title}**")
        st.subheader("🌈 Top Book Tags")
        render_colorful_tags(Book_tags_getter(selected_title))
        render_books(recs_df)
    else:
        st.error("No recommendations found.")

if surprise:
    random_title = df_books['original_title'].sample().iloc[0]
    with st.spinner("Fetching surprise recommendations..."):
        recs_df = get_hybrid_recommendations(random_title, n_recommendations=5)
    if recs_df is not None:
        st.markdown(f"🎁 Surprise book: **{random_title}**")
        st.subheader("🌈 Top Book Tags")
        render_colorful_tags(Book_tags_getter(random_title))
        render_books(recs_df)

# ---- Top Rated by Year ----
st.markdown("---")
st.subheader("🏆 Top Rated Books by Publication Year")
years = sorted(df_books['original_publication_year'].dropna().unique(), reverse=True)
year_choice = st.selectbox("Select Year", options=years)
if st.button("View Top Rated"):
    with st.spinner("Fetching top rated books..."):
        top_books_df = Top_Books_Rated_year(year_choice)
    if not top_books_df.empty:
        st.success(f"Top 5 Rated Books of {year_choice}")
        render_books(top_books_df, show_year=False)
    else:
        st.error(f"No top rated books found for {year_choice}.")
