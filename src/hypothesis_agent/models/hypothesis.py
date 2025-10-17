"""Pydantic models describing hypothesis submission and validation responses."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class RiskAppetite(str, Enum):
    """Discrete risk appetite categories for validation workflows."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class TimeHorizon(BaseModel):
    """Time window associated with the hypothesis."""

    start: date = Field(..., description="Inclusive start date of the hypothesis horizon.")
    end: date = Field(..., description="Inclusive end date of the hypothesis horizon.")


class HypothesisRequest(BaseModel):
    """Inbound contract for hypothesis submissions."""

    user_id: str = Field(..., min_length=1, description="Identifier for the submitting user.")
    hypothesis_text: str = Field(..., min_length=1, description="Natural language hypothesis text.")
    entities: List[str] = Field(default_factory=list, description="Entities referenced in the hypothesis.")
    time_horizon: TimeHorizon = Field(..., description="Time horizon over which the hypothesis should be evaluated.")
    risk_appetite: RiskAppetite = Field(default=RiskAppetite.MODERATE, description="Risk appetite requested by the user.")
    requires_human_review: bool = Field(default=False, description="Whether human approval is required before finalizing.")


class EvidenceReference(BaseModel):
    """Reference to an evidence artifact stored in object storage or external systems."""

    type: str = Field(..., description="Category of the evidence artifact.")
    uri: HttpUrl = Field(..., description="Location of the evidence artifact.")


class ValidationSummary(BaseModel):
    """Summary placeholder for validation outcomes."""

    score: float = Field(..., ge=0.0, le=1.0, description="Validation score normalized between 0 and 1.")
    conclusion: str = Field(..., description="High-level status of the hypothesis validation.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence for the reported conclusion.")
    evidence: List[EvidenceReference] = Field(default_factory=list, description="Evidence artifacts supporting the conclusion.")


class HypothesisResponse(BaseModel):
    """Outbound contract returned to the client after submission."""

    hypothesis_id: UUID = Field(default_factory=uuid4, description="Server-assigned hypothesis identifier.")
    status: str = Field(..., description="Submission status for the hypothesis workflow.")
    validation: ValidationSummary = Field(..., description="Summary of the validation state.")
