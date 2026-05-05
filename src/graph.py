from langgraph.graph import StateGraph, START, END
from src.state import UniversalRAGState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.retriever import retrieve_context
from src.injestion import load_config
from src.providers import get_chat_llm


# Node for the RAG pipeline

def make_retrieve_node(index, config_path):
    def retrieve_node(state):
        config_file = load_config(config_path)
        top_k = config_file['retriever']['top_k']
        results = retrieve_context(index, state['user_question'], top_k)
        return {"retrieved_chunks": results}
    return retrieve_node

def make_rewrite_node(config_path):
    def rewrite_node(state):

        # If there is no history then pass on the question directly
        if not state['messages']:
            return{'user_question':state['user_question']}

        # If there is a history then combine all the above messages and then combine it and then pass them.
        config_file = load_config(config_path)
        llm = get_chat_llm(config_file)


        history = "\n".join([m.content for m in state['messages']])
        
        response = llm.invoke([
            SystemMessage(content="Rewrite the following question to be standalone and clear based on the conversation history. Return only the rewritten question, nothing else."),
            HumanMessage(content=f"History:\n{history}\n\nQuestion:\n{state['user_question']}")
        ])
        
        return {"user_question": response.content}
    return rewrite_node


def make_generate_node(config_path):
    def generate_node(state):
        config_file = load_config(config_path)
        llm = get_chat_llm(config_file)
        
        chunks = "\n\n".join(state['retrieved_chunks'])
        
        # build full message history
        messages = [SystemMessage(content=config_file['prompt']['system'])]
        
        # add previous conversation turns
        messages.extend(state['messages'])
        
        # add current question with context
        messages.append(
            HumanMessage(content=f"Context:\n{chunks}\n\nQuestion:\n{state['user_question']}")
        )
        
        response = llm.invoke(messages)
        
        return {
            'answer': response.content,
            'messages': [
                HumanMessage(content=state['user_question']),
                AIMessage(content=response.content)
            ]
        }
    return generate_node

# Build Graph

def build_graph(index, config_path):
    graph = StateGraph(UniversalRAGState)

    graph.add_node('retrieve', make_retrieve_node(index, config_path))
    graph.add_node('rewrite', make_rewrite_node(config_path))
    graph.add_node('generate', make_generate_node(config_path))

    graph.add_edge(START, 'rewrite')       
    graph.add_edge('rewrite', 'retrieve')  
    graph.add_edge('retrieve', 'generate') 
    graph.add_edge('generate', END)

    return graph.compile()