import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_noise_sections(text):
    blacklist = [
        "acknowledgement", "declaration", "table of contents",
        "certificate", "index", "preface", "abstract"
    ]
    for word in blacklist:
        text = text.replace(word, "")
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9.,!? ]+', '', text)
    return text.strip()

def tokenize_sentences(text):
    return nltk.sent_tokenize(text)

def tokenize_words(text):
    return [w for w in nltk.word_tokenize(text) if w not in stop_words]
