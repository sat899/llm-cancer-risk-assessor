"""FastAPI route handlers."""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from src.schemas import (
    AssessmentRequest,
    AssessmentResponse,
    Citation,
    ChatRequest,
    ChatResponse,
    ChatCitation,
    ChatHistoryResponse,
    ChatHistoryMessage,
)
from src.agent import AssessmentAgent
from src.chat import ChatAgent, get_history, clear_history
from src.tools import get_all_patients

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_agent: Optional[AssessmentAgent] = None
_chat_agent: Optional[ChatAgent] = None


def get_agent() -> AssessmentAgent:
    """Get or initialize the assessment agent."""
    global _agent
    if _agent is None:
        _agent = AssessmentAgent()
    return _agent


def get_chat_agent() -> ChatAgent:
    """Get or initialize the chat agent."""
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = ChatAgent()
    return _chat_agent


# ---------------------------------------------------------------------------
# General endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Clinical Decision Support Agent",
        "version": "2.0.0",
        "status": "operational",
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/patients")
async def list_patients():
    """Get list of all available patient IDs."""
    try:
        patient_ids = get_all_patients()
        return {"patients": patient_ids, "count": len(patient_ids)}
    except Exception as e:
        logger.error(f"Error listing patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Phase 1 — Patient risk assessment
# ---------------------------------------------------------------------------

@router.post("/assess", response_model=AssessmentResponse)
async def assess_patient(request: AssessmentRequest):
    """Assess cancer risk for a patient."""
    try:
        agent = get_agent()
        result = agent.assess(request.patient_id)

        citations = [
            Citation(**citation) for citation in result.get("citations", [])
        ]

        return AssessmentResponse(
            patient_id=result["patient_id"],
            assessment=result["assessment"],
            reasoning=result["reasoning"],
            citations=citations,
            relevant_symptoms=result["relevant_symptoms"],
            confidence=result["confidence"],
        )
    except ValueError as e:
        logger.error(f"Patient not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error assessing patient: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


# ---------------------------------------------------------------------------
# Phase 2 — Conversational chat over NG12 guidelines
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the NG12 guideline chat agent."""
    try:
        agent = get_chat_agent()
        result = agent.chat(
            session_id=request.session_id,
            message=request.message,
            top_k=request.top_k,
        )

        citations = [
            ChatCitation(**c) for c in result.get("citations", [])
        ]

        return ChatResponse(
            session_id=result["session_id"],
            answer=result["answer"],
            citations=citations,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
async def chat_history(session_id: str):
    """Get conversation history for a chat session."""
    history = get_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[ChatHistoryMessage(**m) for m in history],
    )


@router.delete("/chat/{session_id}")
async def chat_delete(session_id: str):
    """Clear conversation history for a chat session."""
    existed = clear_history(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return {"status": "deleted", "session_id": session_id}
