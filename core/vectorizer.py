from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def build_vectorizer(chunks):
    # If dataset is small, relax min_df
    min_df = 1
    max_df = 1.0 if len(chunks) < 10 else 0.9

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=max_df,
        min_df=min_df,
        ngram_range=(1,2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix


def save_vectorizer(vectorizer, tfidf_matrix, path="data/tfidf.pkl"):
    with open(path, "wb") as f:
        pickle.dump((vectorizer, tfidf_matrix), f)


def load_vectorizer(path="data/tfidf.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
