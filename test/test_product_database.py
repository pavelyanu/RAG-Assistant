import os
import sqlite3
from collections.abc import Generator

from pytest import fixture

from rag_assistant.database.product_database import (
    SQLiteProductDatabase,
    get_fake_store_data,
)


class MockSQLiteProductDatabase(SQLiteProductDatabase):
    file = "/tmp/mock_sqlite_product_database_file.db"

    def __init__(self):
        self._connection = sqlite3.connect(self.file)

    def close(self) -> None:
        self._connection.close()
        os.remove(self.file)


@fixture
def mock_sqlite_database_empty() -> Generator[MockSQLiteProductDatabase]:
    database = MockSQLiteProductDatabase()
    yield database
    database.close()


@fixture
def mock_sqlite_database_full(
    mock_sqlite_database_empty,
) -> Generator[MockSQLiteProductDatabase]:
    mock_sqlite_database_empty.create_tables()
    products = get_fake_store_data()
    mock_sqlite_database_empty.insert_products(products)
    yield mock_sqlite_database_empty


def test_sqlite_create_tables(mock_sqlite_database_empty):
    mock_sqlite_database_empty.create_tables()


def test_sqlite_insert_products(mock_sqlite_database_empty):
    mock_sqlite_database_empty.create_tables()
    products = get_fake_store_data()
    mock_sqlite_database_empty.insert_products(products)


def test_sqlite_insert_and_fetch_all(mock_sqlite_database_empty):
    mock_sqlite_database_empty.create_tables()
    products_in = get_fake_store_data()
    mock_sqlite_database_empty.insert_products(products_in)
    products_out = mock_sqlite_database_empty.fetch_all()
    products_in = sorted(products_in, key=lambda x: x.id)
    products_out = sorted(products_out, key=lambda x: x.id)
    for i, o in zip(products_in, products_out, strict=True):
        assert i.id == o.id
        assert i.title == o.title
        assert i.price == o.price
        assert i.description == o.description
        assert i.category == o.category
        assert i.image == o.image
