"""FastAPI route handlers."""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from src.schemas import AssessmentRequest, AssessmentResponse, Citation
from src.agent import AssessmentAgent
from src.tools import get_all_patients

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-loaded agent instance
_agent: Optional[AssessmentAgent] = None


def get_agent() -> AssessmentAgent:
    """Get or initialize the agent instance."""
    global _agent
    if _agent is None:
        _agent = AssessmentAgent()
    return _agent


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Clinical Decision Support Agent",
        "version": "1.0.0",
        "status": "operational"
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
            confidence=result["confidence"]
        )
    except ValueError as e:
        logger.error(f"Patient not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error assessing patient: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")
