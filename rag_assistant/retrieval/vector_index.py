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
    dim: int
    capacity: int
    vectors: np.ndarray
    index: dict[int, str]

    def __init__(self, dim, capacity):
        self.dim = dim
        self.capacity = capacity
        self.vectors = np.zeros((capacity, dim))
        self.index = {}

    def insert(self, data: list[tuple[str, np.ndarray]]) -> None:
        for document, v in data:
            if len(self.index) >= self.capacity:
                raise ValueError("Index is full")
            index = len(self.index)
            self.index[index] = document
            self.vectors[index] = v

    def search(self, v: np.ndarray, k: int) -> list[str]:
        similarity = cosine_similarity(self.vectors, v)
        indices = np.argsort(similarity)[::-1]
        indices = indices[:k]
        results = [self.index[i] for i in indices]
        return results
