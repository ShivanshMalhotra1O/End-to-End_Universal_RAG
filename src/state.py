# State class to store the state of the RAG pipeline

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class UniversalRAGState(TypedDict):
    user_question: str
    retrieved_chunks: list[str]
    answer: str
    messages: Annotated[list, add_messages]  # This means that the messages field is a list of messages, and the add_messages function is used to add messages to the list. 

    