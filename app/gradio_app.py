import gradio as gr
from app.rag_graph import graph
from dotenv import load_dotenv

load_dotenv()

def chat(message, history):
    """
    Chat function for Gradio interface.
    """
    try:
        # The graph expects a dictionary with "messages"
        inputs = {"messages": [message]}
        result = graph.invoke(inputs)
        
        answer = result.get("answer", "No answer generated.")
        context = result.get("context", [])
        
        # Format sources
        sources_text = "\n\n**Retrieved Context:**\n"
        for i, doc in enumerate(context):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            snippet = doc.page_content[:200].replace("\n", " ")
            sources_text += f"{i+1}. **Page {page}**: ...{snippet}...\n"
            
        final_response = f"{answer}\n{sources_text}"
        
        return final_response
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Create the Gradio interface
    demo = gr.ChatInterface(
        fn=chat,
        title="RAG Agentic Chatbot",
        description="Interact with the RAG agent. Answers are based on 'Ebook-Agentic-AI.pdf'.",
        examples=["What is Agentic AI?", "How does RAG work?", "Explain the main concepts"],
    )
    
    # Launch the app
    demo.launch(debug=True, server_name="127.0.0.1", server_port=7860)
