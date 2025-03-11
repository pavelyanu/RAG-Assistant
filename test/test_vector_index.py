from collections.abc import Generator

import numpy as np
import pytest
from pytest import fixture

from rag_assistant.retrieval.vector_index import NumpyVectorIndex


@fixture
def pandas_vector_index() -> Generator[NumpyVectorIndex]:
    yield NumpyVectorIndex(3, 10)


@fixture
def dataset() -> Generator[dict[str, np.ndarray]]:
    data = [
        ("Apple", np.array([1, 1, 1])),
        ("Melon", np.array([-1, -1, 1])),
    ]
    yield data


def test_numpy_insert(pandas_vector_index, dataset):
    pandas_vector_index.insert(dataset)


@pytest.mark.parametrize(
    "query,answer",
    [
        (np.array([1, 1, 1]), "Apple"),
        (np.array([1, 0.5, 1]), "Apple"),
        (np.array([-1, -1, 1]), "Melon"),
        (np.array([-1, -0.3, 1]), "Melon"),
    ],
)
def test_numpy_insert_and_search(pandas_vector_index, dataset, query, answer):
    pandas_vector_index.insert(dataset)
    document = pandas_vector_index.search(query, 1)[0]
    assert document == answer
