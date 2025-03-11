from abc import ABC, abstractmethod

from openai import OpenAI
from tiktoken import Encoding, encoding_for_model


class Embedding(ABC):
    @abstractmethod
    def count_tokens(self, s: str) -> int: ...

    @abstractmethod
    def embed(self, s: str) -> list[float]: ...


class SafeEmbeddingDecorator(Embedding):
    _embedding: Embedding
    _limit: int

    def __init__(self, embedding: Embedding, limit: int):
        self._embedding = embedding
        self._limit = limit

    def count_tokens(self, s: str) -> int:
        return self._embedding.count_tokens(s)

    def embed(self, s: str) -> list[float]:
        if self.count_tokens(s) > self._limit:
            raise ValueError("Token limit exceeded")
        return self._embedding.embed(s)


class OpenAIEmbedding(Embedding):
    _model: str
    _client: OpenAI
    _encoding: Encoding

    def __init__(self, model: str):
        self._model = model
        self._encoding = encoding_for_model(self._model)
        self._client = OpenAI()

    def count_tokens(self, s: str) -> int:
        return len(self._encoding.encode(s))

    def embed(self, s: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._model,
            input=s,
        )
        return response.data[0].embedding
