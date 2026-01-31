import operator
import os
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.vector_store import get_vector_store

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    context: List[Document]
    answer: str

def retrieve(state: AgentState):
    """
    Retrieve documents relevant to the last message.
    """
    print("---RETRIEVE---")
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    vector_store = get_vector_store()
    # Retrieve top 3 documents
    docs = vector_store.similarity_search(query, k=3)
    
    return {"context": docs}

def generate(state: AgentState):
    """
    Generate answer using RAG.
    """
    print("---GENERATE---")
    messages = state["messages"]
    context = state["context"]
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    # Format context
    context_text = "\n\n".join([d.page_content for d in context])
    
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        Question: {question} 
        
        Context: 
        {context} 
        
        Answer:"""
    )
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
    chain = prompt | llm
    
    response = chain.invoke({"question": query, "context": context_text})
    
    return {"answer": response.content}

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

graph = build_graph()
