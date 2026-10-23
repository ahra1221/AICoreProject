"""Deterministic test-only embeddings. These do NOT measure semantic quality."""
import hashlib
import math
from langchain_core.embeddings import Embeddings


class LocalTestEmbeddings(Embeddings):
    def embed_query(self, text):
        vector = [0.0] * 64
        for token in text.casefold().split():
            bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 64
            vector[bucket] += 1
        norm = math.sqrt(sum(value * value for value in vector)) or 1
        return [value / norm for value in vector]
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
