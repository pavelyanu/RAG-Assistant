import os
import sqlite3
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, HttpUrl, TypeAdapter, ValidationError

DATABASE_ENVIRON = "PRODUCT_DATABASE_FILE"


class Product(BaseModel):
    id: int
    title: str
    price: float
    description: str
    category: str
    image: HttpUrl


def get_fake_store_data() -> list[Product] | None:
    response = requests.get("https://fakestoreapi.com/products")
    product_list = TypeAdapter(list[Product])
    try:
        products = product_list.validate_json(response.content)
        return products
    except ValidationError as err:
        print("Validation of fake store api data has failed!")
        print(err)


class ProductDatabase(ABC):
    @abstractmethod
    def create_tables(
        self,
    ): ...

    @abstractmethod
    def insert_products(self, products: list[Product] | Product) -> None: ...

    @abstractmethod
    def fetch_products(self, ids: list[int] | int) -> list[Product]: ...

    @abstractmethod
    def fetch_all(
        self,
    ) -> list[Product]: ...

    @abstractmethod
    def delete_products(self, ids: list[int] | int): ...

    @abstractmethod
    def delete_all(
        self,
    ) -> None: ...

    @abstractmethod
    def close(
        self,
    ) -> None: ...


class SQLiteProductDatabase(ProductDatabase):
    connection: sqlite3.Connection
    table_name: str = "products"

    def __init__(self):
        load_dotenv()
        self.connection = sqlite3.connect(os.environ[DATABASE_ENVIRON])

    def create_tables(self) -> None:
        cur = self.connection.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY,
                title STRING NOT NULL,
                price FLOAT NOT NULL,
                description STR DEFAULT '',
                category STRING NOT NULL,
                image STRING
            )
        """)
        cur.close()

    def insert_products(self, products: list[Product] | Product) -> None:
        if isinstance(products, Product):
            products = [products]

        products = [
            SQLiteProductDatabase._product_model_dump_sqlite(x) for x in products
        ]

        cur = self.connection.cursor()
        cur.executemany(
            f"""
            INSERT INTO {self.table_name} (
                id, title, price, description, category, image
            ) VALUES (
                :id, :title, :price, :description, :category, :image
            )""",
            products,
        )
        self.connection.commit()
        cur.close()

    def fetch_products(self, ids) -> list[Product]:
        raise NotImplementedError()

    def fetch_all(self) -> list[Product]:
        cur = self.connection.cursor()
        cur.execute(
            f"""
            SELECT id, title, price, description, category, image
            FROM {self.table_name}
            """
        )
        products = []
        for row in cur.fetchall():
            products.append(
                Product(
                    id=row[0],
                    title=row[1],
                    price=row[2],
                    description=row[3],
                    category=row[4],
                    image=row[5],
                )
            )
        cur.close()
        return products

    def delete_products(self, ids) -> None:
        raise NotImplementedError()

    def delete_all(self) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _product_model_dump_sqlite(product: Product, *args, **kwargs):
        model = product.model_dump(*args, **kwargs)
        model["image"] = str(model["image"])
        return model
