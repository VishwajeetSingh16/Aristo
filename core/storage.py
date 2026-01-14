import pickle
import os

DATA_DIR = "data"

def save_chunks(chunks, path=os.path.join(DATA_DIR, "chunks.pkl")):
    with open(path, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(path=os.path.join(DATA_DIR, "chunks.pkl")):
    with open(path, "rb") as f:
        return pickle.load(f)

def clear_database():
    files = ["chunks.pkl", "tfidf.pkl"]
    for file in files:
        full_path = os.path.join(DATA_DIR, file)
        if os.path.exists(full_path):
            os.remove(full_path)
