"""
api.py
────────────────────────────────────────────────────────────────────────────────
FastAPI application exposing the churn chatbot as a REST API.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /chat/start          → Start a new conversation session
    POST /chat/message        → Send a message to an existing session
    GET  /chat/status/{id}    → Get session state (features collected so far)
    POST /chat/reset/{id}     → Reset a session
    GET  /health              → Health check
────────────────────────────────────────────────────────────────────────────────
"""

import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot_pipeline import ChurnChatSession, get_missing_fields, answer_followup

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction Chatbot API",
    description=(
        "Conversational API for the marketing team to assess customer churn risk. "
        "Powered by a Random Forest model (AUC 0.85) and a local open-source LLM."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # "null" covers opening chat_tester.html from file://
    allow_origins=["*", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

_sessions: dict[str, ChurnChatSession] = {}


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class StartResponse(BaseModel):
    session_id: str
    message: str


class MessageRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    session_id: str
    reply: str
    prediction_done: bool
    churn_probability: Optional[float] = None
    risk_tier: Optional[str] = None
    missing_fields: list[str] = []


class StatusResponse(BaseModel):
    session_id: str
    collected_features: dict
    missing_fields: list[str]
    prediction_done: bool
    last_prediction: Optional[dict] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_session(session_id: str) -> ChurnChatSession:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found.")
    return _sessions[session_id]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Quick liveness check."""
    return {"status": "ok", "model": "RandomForest (AUC 0.85)", "llm": "groq/llama-3.3-70b-versatile"}


@app.post("/chat/start", response_model=StartResponse, tags=["Chat"])
def start_session():
    """
    Create a new conversation session.
    Returns a session_id that must be included in all subsequent requests.
    """
    session_id = str(uuid.uuid4())
    session = ChurnChatSession()
    greeting = session.start()
    _sessions[session_id] = session
    return StartResponse(session_id=session_id, message=greeting)


@app.post("/chat/message", response_model=MessageResponse, tags=["Chat"])
def send_message(body: MessageRequest):
    """
    Send a user message to the chatbot.

    The chatbot will:
    - Extract customer features from the conversation
    - Ask follow-up questions for missing information
    - Run the churn model once all features are collected
    - Return a plain-English explanation of the result
    """
    session = _get_session(body.session_id)

    if session.prediction_done:
        # Prediction already delivered — answer any follow-up question from the agent
        session._add("user", body.message)
        reply = answer_followup(body.message, session.last_prediction)
        session._add("assistant", reply)
    else:
        reply = session.respond(body.message)

    missing = get_missing_fields(session.collected)

    return MessageResponse(
        session_id=body.session_id,
        reply=reply,
        prediction_done=session.prediction_done,
        churn_probability=(
            session.last_prediction["churn_probability"]
            if session.last_prediction else None
        ),
        risk_tier=(
            session.last_prediction["risk_tier"]
            if session.last_prediction else None
        ),
        missing_fields=missing,
    )


@app.get("/chat/status/{session_id}", response_model=StatusResponse, tags=["Chat"])
def get_status(session_id: str):
    """
    Inspect what features have been collected so far in a session.
    Useful for debugging or building a progress UI.
    """
    session = _get_session(session_id)
    return StatusResponse(
        session_id=session_id,
        collected_features=session.collected,
        missing_fields=get_missing_fields(session.collected),
        prediction_done=session.prediction_done,
        last_prediction=session.last_prediction,
    )


@app.post("/chat/reset/{session_id}", response_model=StartResponse, tags=["Chat"])
def reset_session(session_id: str):
    """Reset a session to start a new customer assessment."""
    session = _get_session(session_id)
    greeting = session.reset()
    return StartResponse(session_id=session_id, message=greeting)
