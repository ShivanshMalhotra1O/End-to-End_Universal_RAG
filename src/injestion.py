# This is the ingestion module for the RAG pipeline

import yaml
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from src.providers import get_llm, get_embedding_model

def load_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)

def build_index(config_path: str) -> VectorStoreIndex:
    config_file = load_config(config_path)

    storage_path = config_file['storage']['path']

    Settings.llm = get_llm(config_file)
    Settings.embed_model = get_embedding_model(config_file)

    if os.path.exists(storage_path):
        print("Loading index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
        return index

    else:
        print("Building fresh index...")
        
        documents = SimpleDirectoryReader(input_files=[config_file['document']['path']]).load_data()

        splitter = SentenceSplitter(
            chunk_size=config_file['document']['chunk_size'],
            chunk_overlap=config_file['document']['chunk_overlap'],
        )

        nodes = splitter.get_nodes_from_documents(documents)

        # filter empty nodes
        nodes = [node for node in nodes if node.text and node.text.strip()]

        # embed one by one to avoid batch bug
        for node in nodes:
            embedding = Settings.embed_model.get_text_embedding(node.text)
            node.embedding = embedding

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=storage_path)
        return index

