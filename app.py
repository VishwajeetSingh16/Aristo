import streamlit as st
import os

os.makedirs("data", exist_ok=True)
os.makedirs("data/pdfs", exist_ok=True)

from core.pdf_loader import load_pdf
from core.preprocessor import clean_text
from core.chunker import chunk_text
from core.vectorizer import build_vectorizer, save_vectorizer, load_vectorizer
from core.storage import save_chunks, load_chunks, clear_database
from core.retriever import retrieve_chunks
from core.answer_builder import build_answer
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_WORDS, MAX_WORDS, TOP_K_CHUNKS


st.set_page_config(page_title="Aristo – Academic RAG System", layout="wide")

st.title("📚 Aristo – Intelligent Academic RAG System")
st.markdown("Upload textbooks, ask questions, and receive grounded academic answers.")

DATA_READY = os.path.exists("data/tfidf.pkl") and os.path.exists("data/chunks.pkl")

# Sidebar
st.sidebar.header("Controls")

if st.sidebar.button("🗑️ Clear Database"):
    clear_database()
    st.sidebar.success("Database cleared. Upload new PDFs to rebuild.")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF textbooks",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("⚙️ Build Knowledge Base"):
    if not uploaded_files:
        st.sidebar.error("Upload at least one PDF.")
    else:
        all_chunks = []
        for file in uploaded_files:
            with open(f"data/pdfs/{file.name}", "wb") as f:
                f.write(file.read())

            text = load_pdf(f"data/pdfs/{file.name}")
            cleaned = clean_text(text)
            chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)
            all_chunks.extend(chunks)

        vectorizer, tfidf_matrix = build_vectorizer(all_chunks)

        save_chunks(all_chunks)
        save_vectorizer(vectorizer, tfidf_matrix)

        st.sidebar.success("Knowledge base built successfully.")

# Main Query Area
st.subheader("Ask Your Question")

query = st.text_area("Enter your academic question:", height=100)

word_limit = st.slider(
    "Answer Length (words)",
    min_value=500,
    max_value=800,
    value=650,
    step=50
)

if st.button("🔍 Generate Answer"):
    if not DATA_READY:
        st.error("Build the knowledge base first by uploading PDFs.")
    elif not query.strip():
        st.error("Enter a question.")
    else:
        chunks = load_chunks()
        vectorizer, tfidf_matrix = load_vectorizer()

        results = retrieve_chunks(query, vectorizer, tfidf_matrix, chunks, TOP_K_CHUNKS)
        answer = build_answer(results, query, min_words=500, max_words=word_limit)


        st.subheader("📖 Answer")
        st.write(answer)

        with st.expander("🔎 Retrieved Context (Debug)"):
            for r in results:
                st.markdown(f"**Score:** {r['score']:.4f}")
                st.write(r["chunk"][:500] + "...")
