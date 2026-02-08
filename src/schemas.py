"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Phase 1 — Patient risk assessment
# ---------------------------------------------------------------------------

class AssessmentRequest(BaseModel):
    """Request model for patient assessment."""
    patient_id: str = Field(..., description="Patient identifier (e.g., PT-101)")


class Citation(BaseModel):
    """Citation model for guideline references."""
    page_number: int
    section: str
    content: str
    relevance_score: float


class AssessmentResponse(BaseModel):
    """Response model for patient assessment."""
    patient_id: str
    assessment: str = Field(
        ..., 
        description="Risk assessment: 'Urgent Referral', 'Urgent Investigation', or 'Routine'"
    )
    reasoning: str
    citations: List[Citation]
    relevant_symptoms: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Phase 2 — Conversational chat over NG12 guidelines
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request model for a chat message."""
    session_id: str = Field(..., description="Client-generated session identifier")
    message: str = Field(..., description="User question about NG12 guidelines")
    top_k: int = Field(5, ge=1, le=20, description="Number of guideline chunks to retrieve")


class ChatCitation(BaseModel):
    """Citation returned with a chat answer."""
    source: str = Field("NG12 PDF", description="Source document")
    page: int = Field(..., description="Page number in the NG12 PDF")
    chunk_id: str = Field(..., description="Identifier for the retrieved chunk")
    excerpt: str = Field(..., description="Short excerpt from the guideline")


class ChatResponse(BaseModel):
    """Response model for a chat message."""
    session_id: str
    answer: str
    citations: List[ChatCitation]


class ChatHistoryMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatHistoryResponse(BaseModel):
    """Response model for GET /chat/{session_id}/history."""
    session_id: str
    messages: List[ChatHistoryMessage]
