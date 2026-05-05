# This is the retriever module for the RAG pipeline, this will be used to retrieve the relevant chunks from the index

from llama_index.core import VectorStoreIndex

def get_retriever(index: VectorStoreIndex, top_k: int):
    return index.as_retriever(similarity_top_k=top_k)

def retrieve_context(index: VectorStoreIndex, query: str, top_k:int) -> list[str]:
    retriever = get_retriever(index, top_k)
    nodes = retriever.retrieve(query)
    return [node.text for node in nodes]