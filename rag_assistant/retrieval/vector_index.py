from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):
    @abstractmethod
    def insert(self, data: list[tuple[str, np.ndarray]]) -> None: ...
    @abstractmethod
    def search(self, v: np.ndarray, k: int) -> list[str]: ...


def cosine_similarity(data: np.ndarray, query: np.ndarray) -> np.ndarray:
    similarity = np.dot(data, query)
    epsilon = 1e-6
    data_norm = np.linalg.norm(data, axis=1) + epsilon
    query_norm = np.linalg.norm(query) + epsilon
    similarity /= data_norm * query_norm
    return similarity


class NumpyVectorIndex(VectorIndex):
    _dim: int
    _capacity: int
    _vectors: np.ndarray
    _index: dict[int, str]

    def __init__(self, dim, capacity):
        self._dim = dim
        self._capacity = capacity
        self._vectors = np.zeros((capacity, dim))
        self._index = {}

    def insert(self, data: list[tuple[str, np.ndarray]]) -> None:
        for document, v in data:
            if len(self._index) >= self._capacity:
                raise ValueError("Index is full")
            index = len(self._index)
            self._index[index] = document
            self._vectors[index] = v

    def search(self, v: np.ndarray, k: int) -> list[str]:
        similarity = cosine_similarity(self._vectors, v)
        indices = np.argsort(similarity)[::-1]
        indices = indices[:k]
        results = [self._index[i] for i in indices]
        return results
