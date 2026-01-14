import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_chunks(query, vectorizer, tfidf_matrix, chunks, top_k=10):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "score": similarities[idx]
        })
    return results
