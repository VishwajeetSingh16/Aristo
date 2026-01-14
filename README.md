# Aristo – Intelligent Academic RAG System

Aristo is a CPU-based, fully offline, academic Retrieval-Augmented Generation (RAG) system.  
It allows students to upload textbooks in PDF format, ask academic questions, and receive structured, exam-oriented answers strictly grounded in the uploaded material.

This project uses:
- Classical NLP
- TF-IDF / BM25 retrieval
- Extractive answer synthesis
- No pretrained LLMs
- No GPU dependency

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/aristo.git
cd aristo


python -m venv aristo
.\aristo\Scripts\activate


pip install -r requirements.txt

run:
python

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
exit()
```

## 2. Run the frontend
streamlit run app.py
