from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel
from app.rag_graph import graph
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Agentic Chatbot")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG Agentic Chatbot is running"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Initial invoke with a simple string message for now
        # In a full valid implementation, we'd use LangChain Message objects
        result = graph.invoke({"messages": [request.message]})
        return {"response": result["messages"][-1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
