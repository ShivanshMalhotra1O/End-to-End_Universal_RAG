# This is the providers module - acts as a factory for LLM and embedding models
# When you want to switch providers, only rag_config.yaml needs to change

import os

def get_llm(config: dict):
    provider = config["provider"]
    model = config["llm"]["model"]

    if provider == "gemini":
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(
            model=model,
            api_key=os.environ.get("GOOGLE_API_KEY")
        )

    elif provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: gemini, openai")


def get_embedding_model(config: dict):
    provider = config["provider"]
    model = config["embeddings"]["model"]

    if provider == "gemini":
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        return GoogleGenAIEmbedding(
            model=model,
            api_key=os.environ.get("GOOGLE_API_KEY"),
            embed_batch_size=5
        )

    elif provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: gemini, openai")


def get_chat_llm(config: dict):
    provider = config["provider"]
    model = config["llm"]["model"]

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: gemini, openai")