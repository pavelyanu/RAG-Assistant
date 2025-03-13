from abc import ABC, abstractmethod

from rag_assistant.llm.chat import Chat, ChatMessage
from rag_assistant.retrieval.embedding import Embedding
from rag_assistant.retrieval.vector_index import VectorIndex


def chat_history_to_text(chat_history: list[ChatMessage]):
    return "\n\n".join([f"{m.role}: {m.content}" for m in chat_history])


class Agent(ABC):
    @abstractmethod
    def respond(self, messages: ChatMessage) -> ChatMessage: ...

    @abstractmethod
    def reset(self) -> None: ...


class SimpleChatBot(Agent):
    _chat: Chat
    _history: list[ChatMessage]

    def __init__(self, chat: Chat):
        self._chat = chat
        self._history = [
            ChatMessage(
                role="developer",
                content="You are an assistant for an e-commerce web site.",
            )
        ]

    def respond(self, message: ChatMessage) -> ChatMessage:
        self._history.append(message)
        answer = self._chat.chat(self._history)
        self._history.append(answer)
        return answer

    def reset(self) -> None:
        self._history.clear()


class SimpleRAGAgent(Agent):
    _chat: Chat
    _history: list[ChatMessage]
    _vector_index: VectorIndex
    _k: int
    _embedding: Embedding
    _search_decision_prompt = """
        You are an assistant for an e-commerce web site.
        Consider the last message of the conversation:
        ---
        {0}
        ---

        Do we need to search the database of products to answer this query?
        
        Answer "yes" or "no" in lower case.
        """
    _construct_query_prompt = """
        You are an assistant for an e-commerce web site.
        Consider the following conversation:
        ---
        {0}
        ---
        Construct a query that will be later embedded and used to search for
        relevant results in vector database of product descriptions. Here is an
        example of a product description:
        ---
        Your perfect pack for everyday use and walks in the forest. Stash your laptop (up to 15 inches) in the padded sleeve, your everyday
        ---
    """
    _user_question_with_search_results_prompt = """
        You are an assistant for an e-commerce web site.
        In addition to the last question from the user:
        ---
        {0}
        ---
        Here are the search results from the database of products:
        ---
        {1}
        ---
    """

    def __init__(
        self, chat: Chat, vector_index: VectorIndex, embedding: Embedding, k=2
    ):
        self._history = [
            ChatMessage(
                role="developer",
                content="You are an assistant for an e-commerce web site.",
            )
        ]
        self._chat = chat
        self._vector_index = vector_index
        self._embedding = embedding
        self._k = k

    def respond(self, message: ChatMessage) -> ChatMessage:
        if self.decide_to_search(message):
            search_results = self.search(message)
            extended_message = ChatMessage(
                role="user",
                content=self._user_question_with_search_results_prompt.format(
                    message.content,
                    "\n\n".join(search_results[: self._k]),
                ),
            )
            answer = self._chat.chat(self._history + [extended_message])
        else:
            answer = self._chat.chat([message])

        self._history.append(message)
        self._history.append(answer)
        return answer

    def decide_to_search(self, message: ChatMessage) -> bool:
        answer = self._chat.chat(
            [
                ChatMessage(
                    role="user",
                    content=self._search_decision_prompt.format(message.content),
                )
            ]
        )
        return "yes" in answer.content.lower()

    def search(self, message: ChatMessage) -> list[str]:
        query = self._chat.chat(
            [
                ChatMessage(
                    role="user",
                    content=self._construct_query_prompt.format(message.content),
                )
            ]
        )
        query_embedding = self._embedding.embed(query.content)
        return self._vector_index.search(query_embedding, self._k)

    def reset(self) -> None:
        self._history.clear()
