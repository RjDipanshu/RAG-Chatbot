# precise RAG Agentic Chatbot

A precise Retrieval-Augmented Generation (RAG) chatbot using LangChain, LangGraph, and Pinecone, designed to answer questions based strictly on a provided PDF document with accurate page citations.

## Features

- **Strict Grounding:** Answers are derived *only* from the provided document context.
- **Accurate Citations:** Provides exact printed page numbers from the PDF footer (not just file index).
- **Agentic Workflow:** Uses LangGraph for a structured retrieval and generation flow.
- **Vector Search:** Utilizes Pinecone for efficient similarity search.
- **Interactive UI:** Features a chat interface built with Gradio and accessible via a standard API.

## Prerequisites

- **Python 3.10+**
- **Pinecone Account:** You need an API key and an index name.
- **Groq API Key:** For the LLM (Llama 3.1).
- **HuggingFace Token:** For embeddings (optional, usually required for gated models).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rag-agentic-chatbot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Create a `.env` file in the `app` directory (or root, depending on where you run commands) based on the following template:

    ```env
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX_NAME=your_index_name
    GROQ_API_KEY=your_groq_api_key
    # Add other keys if necessary (e.g., OPENAI_API_KEY if you switch models)
    ```

2.  **PDF Data:**
    - Place your PDF file in the `data/` directory.
    - Default expected file: `data/Ebook-Agentic-AI.pdf`.
    - If you change the filename, update `app/ingest.py`.

## Usage

### 1. Ingest Data

Before running the chatbot, you must process the PDF to extract text and generate embeddings. This step also extracts the printed page numbers for citations.

```bash
python app/ingest.py
```

### 2. Run the Application

Start the FastAPI server, which mounts the Gradio interface.

```bash
uvicorn app.main:app --reload
```

- The API will be available at `http://127.0.0.1:8000`.
- The Chat UI will be accessible continuously.

## Project Structure

- `app/ingest.py`: Handles PDF loading, page number extraction, splitting, and uploading to Pinecone.
- `app/rag_graph.py`: Defines the LangGraph workflow (Retrieve -> Generate) and the system prompt.
- `app/gradio_app.py`: Sets up the Gradio user interface.
- `app/vector_store.py`: Manages the Pinecone connection.
- `app/main.py`: Entry point for the FastAPI application.

