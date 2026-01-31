import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

def get_vector_store():
    """
    Returns a PineconeVectorStore instance using HuggingFace embeddings.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not pinecone_api_key or not index_name:
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in .env")

    # Use a small, efficient local model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
