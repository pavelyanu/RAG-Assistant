import streamlit as st
from dotenv import load_dotenv
from streamlit.delta_generator import DeltaGenerator

from rag_assistant.database.product_database import get_fake_store_data
from rag_assistant.llm.agent import Agent, SimpleChatBot, SimpleRAGAgent
from rag_assistant.llm.chat import ChatMessage, OpenAIChat, SafeChatDecorator
from rag_assistant.retrieval.embedding import (
    Embedding,
    OpenAIEmbedding,
    SafeEmbeddingDecorator,
)
from rag_assistant.retrieval.vector_index import NumpyVectorIndex, VectorIndex


def simple_chat_bot() -> SimpleChatBot:
    load_dotenv()
    model = "gpt-4o-mini-2024-07-18"
    chat = OpenAIChat(model=model)
    chat = SafeChatDecorator(chat, limit=100)
    return SimpleChatBot(chat)


def populate_vector_index(vector_index: VectorIndex, embedding: Embedding):
    products = get_fake_store_data()
    products = [
        (str(product), embedding.embed(product.description)) for product in products
    ]
    vector_index.insert(products)


def simple_rag_agent() -> SimpleRAGAgent:
    load_dotenv()

    chat_model = "gpt-4o-mini-2024-07-18"
    chat = OpenAIChat(model=chat_model)
    chat = SafeChatDecorator(chat, limit=1000)

    embedding_model = "text-embedding-3-small"
    embedding_dim = 100
    embedding = OpenAIEmbedding(model=embedding_model, dim=100)
    embedding = SafeEmbeddingDecorator(embedding, limit=1000)

    vector_index = NumpyVectorIndex(dim=embedding_dim, capacity=100)
    populate_vector_index(vector_index, embedding)

    return SimpleRAGAgent(chat, vector_index, embedding, k=2)


@st.cache_resource
def agent() -> Agent:
    return simple_rag_agent()


@st.cache_resource
def messages() -> list[ChatMessage]:
    return []


def clear_history():
    messages().clear()
    agent().reset()


def display_messages(container: DeltaGenerator):
    for message in messages():
        with container.chat_message(message.role):
            st.markdown(message.content)


def display_assistant(assistant_container: DeltaGenerator):
    with assistant_container:
        st.button("Clear conversation", on_click=clear_history)

    chat_container = assistant_container.container(border=True)
    display_messages(chat_container)

    if prompt := assistant_container.chat_input("Enter your message..."):
        user_message = ChatMessage(role="user", content=prompt)
        messages().append(user_message)

        with chat_container.chat_message("user"):
            st.markdown(prompt)

        with chat_container.chat_message("assistant"):
            response = agent().respond(user_message)
            st.markdown(response.content)

        messages().append(response)


def main():
    st.title("Shopping Assistant")
    display_assistant(st.container())


main()
