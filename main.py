"""
FastAPI service exposing:
  GET  /health -> {"status": "ok"}
  POST /chat   -> {"reply": str, "recommendations": [...], "end_of_conversation": bool}

Stateless: every /chat call receives the full conversation history and
returns the next turn. No server-side session storage.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

import chat_logic

app = FastAPI(title="SHL Assessment Recommendation Agent")


# --- Request/response schemas ---

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        result = chat_logic.handle_chat(messages)
        return result
    except Exception as e:
        import traceback
        print("=== /chat ERROR ===")
        traceback.print_exc()
        return {
            "reply": "I ran into a temporary issue processing that. Could you try again?",
            "recommendations": [],
            "end_of_conversation": False,
        }