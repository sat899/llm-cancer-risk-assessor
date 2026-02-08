"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import List


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
