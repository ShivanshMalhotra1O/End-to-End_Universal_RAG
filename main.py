from dotenv import load_dotenv
import os
from src.injestion import build_index
from src.graph import build_graph

load_dotenv()

config_path = os.getenv("CONFIG_PATH")

print("Loading and indexing document... please wait")
index = build_index(config_path)
print("Index built successfully!")
print("Building graph...")
graph = build_graph(index, config_path)
print("Ready! Ask your questions.\n")

chat_history = []

def find_in_history(question:str , history: list) -> str | None:

    for i,message in enumerate(history):
        if message.type == "human" and message.content.lower() == question.lower():
            # next message is the AI answer
            if i + 1 < len(history):
                return history[i + 1].content
    return None


while(True):

    user_query = input("Enter the question you want to ask (Type exit to close):")

    if user_query.lower() in ['quit', 'exit']:
        break

    else:

        cached = find_in_history(user_query,chat_history)

        if cached:
            print("Retrieved from history:")
            print(cached)
            
        else:
            result = graph.invoke({
                "user_question": user_query,
                "retrieved_chunks": [],
                "answer": "",
                "messages": chat_history
            })


            print(result['answer'])
            chat_history = result['messages']