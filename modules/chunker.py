import os
import json
from typing import List, Tuple
import nltk
import spacy

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")


class TextChunker:
    def __init__(self, method: str = "spacy", max_words: int = 100):
        """
        :param method: "nltk" or "spacy"
        :param max_words: Max words per chunk
        """
        self.method = method
        self.max_words = max_words

    def load_json_folder(self, folder: str) -> List[Tuple[str, str]]:
        """Load and chunk all JSON files from a folder."""
        chunks = []
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                with open(os.path.join(folder, filename), "r") as f:
                    try:
                        raw_data = json.load(f)
                        flat_items = self._flatten_nested_list(raw_data)

                        for i, item in enumerate(flat_items):
                            if isinstance(item, dict):
                                text = item.get("data", "")
                                chunk_id = item.get("id") or item.get("title") or f"{filename}_{i}"
                                if text.strip():
                                    chunks.extend(self.chunk_text(text, prefix=chunk_id))
                            elif isinstance(item, str):
                                chunks.extend(self.chunk_text(item, prefix=f"{filename}_{i}"))

                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
        return chunks

    def _flatten_nested_list(self, data):
        """Recursively flatten nested lists into a flat list of strings or dicts."""
        flat = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    flat.extend(self._flatten_nested_list(item))
                else:
                    flat.append(item)
        else:
            flat.append(data)
        return flat


    def chunk_text(self, text: str, prefix="chunk") -> List[Tuple[str, str]]:
        """Chunk a single string of text into smaller segments."""
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        word_count = 0
        chunk_id = 0

        for sent in sentences:
            words = sent.split()
            if word_count + len(words) > self.max_words:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((f"{prefix}_{chunk_id}", chunk_text))
                    chunk_id += 1
                    current_chunk = []
                    word_count = 0
            current_chunk.extend(words)
            word_count += len(words)

        if current_chunk:
            chunks.append((f"{prefix}_{chunk_id}", " ".join(current_chunk)))
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences based on method."""
        if self.method == "nltk":
            return nltk.sent_tokenize(text)
        elif self.method == "spacy":
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            raise ValueError("Unsupported method. Choose 'nltk' or 'spacy'.")
