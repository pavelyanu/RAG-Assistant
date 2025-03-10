from pytest import fixture

from rag_assistant.retrieval.embedding import OpenAIEmbedding, SafeEmbeddingDecorator


@fixture
def openai_embedding():
    embedding = OpenAIEmbedding(model="text-embedding-3-small")
    embedding = SafeEmbeddingDecorator(embedding, limit=1000)
    yield embedding


@fixture
def string():
    yield "Hello, World!"


def test_count_tokens(openai_embedding, string):
    token_count = openai_embedding.count_tokens(string)
    assert token_count is not None
    assert token_count > 1


def test_embed(openai_embedding, string):
    embedding = openai_embedding.embed(string)
    assert embedding is not None
    assert isinstance(embedding, list)
    assert isinstance(embedding[0], float)
