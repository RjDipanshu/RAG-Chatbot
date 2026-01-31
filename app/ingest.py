import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

def ingest_pdf():
    pdf_path = os.path.join(os.getcwd(), "data", "Ebook-Agentic-AI.pdf")
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print("Initializing Vector Store...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not pinecone_api_key or not index_name:
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in .env")
    
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Adding documents to Pinecone (in batches)...")
    
    batch_size = 50
    total_chunks = len(splits)
    
    for i in range(0, total_chunks, batch_size):
        batch = splits[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...")
        
        try:
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=index_name,
                pinecone_api_key=pinecone_api_key
            )
            time.sleep(1) # Small pause just to be safe
        except Exception as e:
            print(f"Error processing batch {i}: {e}")

    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_pdf()
