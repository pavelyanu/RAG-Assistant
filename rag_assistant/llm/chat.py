from abc import ABC, abstractmethod

from openai import OpenAI
from pydantic import BaseModel
from tiktoken import Encoding, encoding_for_model


class ChatMessage(BaseModel):
    role: str
    content: str


class Chat(ABC):
    @abstractmethod
    def chat(self, messages: list[ChatMessage]) -> ChatMessage: ...
    @abstractmethod
    def count_tokens(self, messages: list[ChatMessage]) -> int: ...


class SafeChatDecorator(Chat):
    _chat: Chat
    _limit: int

    def __init__(self, chat: Chat, limit: int):
        self._chat = chat
        self._limit = limit

    def chat(self, messages: list[ChatMessage]) -> ChatMessage:
        if self.count_tokens(messages) > self._limit:
            raise ValueError("Token limit exceeded")
        return self._chat.chat(messages)

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        return self._chat.count_tokens(messages)


class OpenAIChat(Chat):
    _model: str
    _client: OpenAI
    _encoding: Encoding

    def __init__(self, model: str):
        self._model = model
        self._encoding = encoding_for_model(self._model)
        self._client = OpenAI()

    def chat(self, messages: list[ChatMessage]) -> ChatMessage:
        messages = [message.model_dump() for message in messages]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        message = response.choices[0].message
        return ChatMessage(role=message.role, content=message.content)

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        text = " ".join([m.content for m in messages])
        return len(self._encoding.encode(text))
