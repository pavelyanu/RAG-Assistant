from collections.abc import Generator

from dotenv import load_dotenv
from pytest import fixture

from rag_assistant.llm.chat import ChatMessage, OpenAIChat, SafeChatDecorator


@fixture
def openai_chat() -> Generator[SafeChatDecorator]:
    load_dotenv()
    model = "gpt-4o-mini-2024-07-18"
    chat = OpenAIChat(model=model)
    chat = SafeChatDecorator(chat, limit=100)
    yield chat


@fixture
def chat_messages() -> Generator[list[ChatMessage]]:
    messages = [
        ChatMessage(role="system", content="Extract the event information."),
        ChatMessage(
            role="user",
            content="Alice and Bob are going to a science fair on Friday.",
        ),
    ]
    yield messages


def test_openai_count_tokens(openai_chat, chat_messages):
    count = openai_chat.count_tokens(chat_messages)
    assert count is not None
    assert isinstance(count, int)
    assert count > 5
    assert count < 50


def test_openai_chat(openai_chat, chat_messages):
    response = openai_chat.chat(chat_messages)
    assert response is not None
    assert isinstance(response, ChatMessage)
    assert response.role == "assistant"
    assert response.content != ""
