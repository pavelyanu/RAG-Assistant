from abc import ABC, abstractmethod

from openai import OpenAI
from tiktoken import Encoding, encoding_for_model


class Embedding(ABC):
    @abstractmethod
    def count_tokens(self, s: str) -> int: ...

    @abstractmethod
    def embed(self, s: str) -> list[float]: ...


class SafeEmbeddingDecorator(Embedding):
    embedding: Embedding
    limit: int

    def __init__(self, embedding: Embedding, limit: int):
        self.embedding = embedding
        self.limit = limit

    def count_tokens(self, s: str) -> int:
        return self.embedding.count_tokens(s)

    def embed(self, s: str) -> list[float]:
        if self.count_tokens(s) > self.limit:
            raise ValueError("Token limit exceeded")
        return self.embedding.embed(s)


class OpenAIEmbedding(Embedding):
    model: str
    client: OpenAI
    encoding: Encoding

    def __init__(self, model: str):
        self.model = model
        self.encoding = encoding_for_model(self.model)
        self.client = OpenAI()

    def count_tokens(self, s: str) -> int:
        return len(self.encoding.encode(s))

    def embed(self, s: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=s,
        )
        return response.data[0].embedding
