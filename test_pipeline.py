from core.pdf_loader import load_pdf
from core.preprocessor import clean_text
from core.chunker import chunk_text
from core.vectorizer import build_vectorizer
from core.retriever import retrieve_chunks
from core.answer_builder import build_answer

# Load sample PDF
text = load_pdf("D:\Vishwajeet\Projects\Aristo\data\pdf\ProjectReport_Aristo_Final.pdf")
cleaned = clean_text(text)
chunks = chunk_text(cleaned)

vectorizer, tfidf_matrix = build_vectorizer(chunks)

query = "What is Aristo about?"
results = retrieve_chunks(query, vectorizer, tfidf_matrix, chunks, top_k=15)

answer = build_answer(results)

print(answer)
