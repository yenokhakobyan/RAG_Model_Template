
import sqlite3  # Add SQLite for local database storage
from modules.chunker import TextChunker
from modules.embedder import TextEmbedder
from modules.indexer import FaissIndexer
from modules.retriever import Retriever
from modules.rag_engine import RAGEngine
import numpy as np
from modules.evaluator import Evaluator
import pickle  # For saving and loading the model
import faiss  # Import FAISS for vector indexing

class FaissVectorStore:
    """
    A custom vector store using FAISS for indexing and retrieval.
    """
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)  # L2 distance-based FAISS index
        self.chunk_ids = []
        self.chunk_texts = []

    def add(self, embeddings, chunk_ids, chunk_texts):
        """
        Add embeddings and metadata to the FAISS index.
        """
        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
        self.chunk_texts.extend(chunk_texts)

    def save(self, index_path="faiss_index.bin", metadata_path="faiss_metadata.pkl"):
        """
        Save the FAISS index and metadata to files.
        """
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "chunk_texts": self.chunk_texts}, f)
        print(f"✅ FAISS index saved to {index_path}")
        print(f"✅ Metadata saved to {metadata_path}")

    def load(self, index_path="faiss_index.bin", metadata_path="faiss_metadata.pkl"):
        """
        Load the FAISS index and metadata from files.
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        self.chunk_ids = metadata["chunk_ids"]
        self.chunk_texts = metadata["chunk_texts"]
        print(f"📂 FAISS index loaded from {index_path}")
        print(f"📂 Metadata loaded from {metadata_path}")

def main():
    data_dir = "/Users/yenokhakobyan/RAG_Model_Template/data"

    print("🔍 Loading and chunking data...")
    chunker = TextChunker(method="spacy", max_words=100)
    chunks = chunker.load_json_folder(data_dir)
    chunk_ids, chunk_texts = zip(*chunks)

    chunk_ids, chunk_texts = zip(*chunks)

    print("📐 Embedding chunks...")
    embedder = TextEmbedder()
    embeddings = embedder.encode(chunk_texts).astype("float32")  # FAISS requires float32

    # Initialize and add data to the FaissVectorStore
    vector_store = FaissVectorStore(dim=embeddings.shape[1])
    vector_store.add(embeddings, chunk_ids, chunk_texts)

    # Save the FAISS index and metadata
    vector_store.save()

    # Optionally, load the FAISS index and metadata (for demonstration purposes)
    # vector_store.load()

    retriever = Retriever(embedder, vector_store.index, chunk_texts, chunk_ids)
    rag = RAGEngine(model="llava:7b")

    print("🧠 RAG system ready. Ask me anything!\n")

    while True:
        query = input("❓ Your question (or 'exit'): ")
        if query.lower() == "exit":
            break
        top_chunks = retriever.retrieve(query, top_k=3)
        prompt = rag.build_prompt(top_chunks, query)
        response = rag.query(prompt)
        print("\n🦙 LLaMA (Markdown output):\n")
        print(response.strip())
        print("\n" + "-" * 60 + "\n")
        

    evaluator = Evaluator()  # init once at top of main()

    # After showing response
    feedback = input("⭐️ Do you have a reference answer to evaluate? (y/n): ")
    if feedback.lower().strip() == "y":
        reference = input("🔖 Paste reference answer:\n")
        score = evaluator.similarity_score(response.strip(), reference)
        print(f"\n📊 Similarity score (0–1): {score:.4f}")
        if score > 0.8:
            print("✅ Great alignment!")
        elif score > 0.5:
            print("⚠️ Partial match – could be improved.")
        else:
            print("❌ Low similarity. LLaMA may have hallucinated.")


if __name__ == "__main__":
    main()
