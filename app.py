import streamlit as st
import numpy as np
from gensim.models import Word2Vec, FastText

st.set_page_config(
    page_title="NLP Embedding Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NLP Embedding Explorer")
st.write("Explore Word2Vec and FastText embeddings.")

# Sample corpus
sentences = [
    ["india", "is", "a", "great", "country"],
    ["india", "has", "many", "scientists"],
    ["vikram", "sarabhai", "was", "a", "scientist"],
    ["war", "needs", "strength"],
    ["freedom", "is", "important"],
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "uses", "neural", "networks"]
]

# Sidebar
st.sidebar.header("Model Settings")

embedding_type = st.sidebar.selectbox(
    "Embedding Model",
    ["Word2Vec", "FastText"]
)

vector_size = st.sidebar.slider(
    "Vector Size",
    50,
    300,
    100
)

window = st.sidebar.slider(
    "Window Size",
    2,
    10,
    5
)

# Train Model
if embedding_type == "Word2Vec":
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4
    )
else:
    model = FastText(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4
    )

st.success(f"{embedding_type} model trained successfully!")

# Vocabulary
st.subheader("Vocabulary")

st.write(model.wv.index_to_key)

# Search Word
st.subheader("Search Word")

word = st.text_input("Enter a word", "india")

if st.button("Find Similar Words"):

    if word in model.wv:

        st.success(f"Word '{word}' found.")

        st.subheader("Vector")

        st.write(model.wv[word])

        st.subheader("Most Similar Words")

        similar = model.wv.most_similar(word)

        for w, score in similar:
            st.write(f"**{w}** : {score:.4f}")

    else:

        st.error("Word not found in vocabulary.")

# Word Similarity
st.subheader("Compare Two Words")

col1, col2 = st.columns(2)

with col1:
    w1 = st.text_input("Word 1", "india")

with col2:
    w2 = st.text_input("Word 2", "country")

if st.button("Calculate Similarity"):

    if w1 in model.wv and w2 in model.wv:

        score = model.wv.similarity(w1, w2)

        st.metric("Cosine Similarity", round(score, 4))

    else:

        st.warning("One or both words are missing.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit + Gensim")