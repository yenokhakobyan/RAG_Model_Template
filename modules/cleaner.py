import os
import json
from typing import List, Tuple
import nltk
import spacy
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

def load_and_chunk_json(folder: str, method="nltk", max_chunk_words=100) -> List[Tuple[str, str]]:
    chunks = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    text = item.get("data") or item.get("text", "")
                    chunk_id = item.get("id") or f"{file}_{i}"
                    if text:
                        chunks.extend(chunk_text(text, method, max_chunk_words, chunk_id))
    return chunks

def chunk_text(text, method="nltk", max_chunk_words=100, prefix="chunk") -> List[Tuple[str, str]]:
    sentences = []
    if method == "spacy":
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    else:
        sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    word_count = 0
    chunk_id = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > max_chunk_words:
            if current_chunk:
                chunks.append((f"{prefix}_{chunk_id}", " ".join(current_chunk)))
                chunk_id += 1
                current_chunk = []
                word_count = 0
        current_chunk.extend(words)
        word_count += len(words)

    if current_chunk:
        chunks.append((f"{prefix}_{chunk_id}", " ".join(current_chunk)))
    return chunks
