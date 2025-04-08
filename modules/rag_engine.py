import subprocess

class RAGEngine:
    def __init__(self, model="llama3"):
        self.model = model

    def build_prompt(self, context_chunks: list, question: str) -> str:
        context_md = "\n\n".join([f"- {c}" for c in context_chunks])
        return f"""You are an expert assistant. Use the following **context** to answer the **question**.

### Context:
{context_md}

### Question:
{question}

### Answer (in Markdown):"""

    def query(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8")
