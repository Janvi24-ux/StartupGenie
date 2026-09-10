"""
Backend – Pydantic Models
Request and response schemas for the API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────

class StartupInput(BaseModel):
    """User-provided startup details."""
    idea: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Description of the startup idea.",
        examples=["An AI-powered app that connects local farmers directly with urban consumers."],
    )
    industry: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Industry or sector of the startup.",
        examples=["AgriTech"],
    )
    target_customer: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Primary target customer description.",
        examples=["Urban households aged 25–45 who prefer fresh, organic produce."],
    )
    location: str = Field(
        default="India",
        max_length=100,
        description="Country or region where the startup will operate.",
        examples=["India"],
    )
    budget: str = Field(
        ...,
        max_length=200,
        description="Available startup budget (amount and currency).",
        examples=["INR 10 lakh"],
    )
    stage: str = Field(
        default="Idea Stage",
        max_length=100,
        description="Current stage of the startup.",
        examples=["Idea Stage", "MVP Stage", "Early Traction", "Growth Stage"],
    )


# ──────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────

class RetrievedContext(BaseModel):
    """A single retrieved knowledge-base chunk."""
    content: str
    source: str
    score: float


class LabelCounts(BaseModel):
    """Counts of each label type in the generated blueprint."""
    retrieved_facts: int
    estimates: int
    assumptions: int
    recommendations: int


class ReadinessScore(BaseModel):
    """Startup readiness score breakdown."""
    idea_clarity: int = 0
    market_opportunity: int = 0
    technical_feasibility: int = 0
    business_model_strength: int = 0
    team_resource_readiness: int = 0
    regulatory_readiness: int = 0
    overall: int = 0


class BlueprintResponse(BaseModel):
    """Full startup blueprint response."""
    startup_input: StartupInput
    blueprint_raw: str
    sections: Dict[str, str]
    retrieved_context: List[RetrievedContext]
    readiness_score: ReadinessScore
    label_counts: LabelCounts
    model_used: str
    rag_chunks_retrieved: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    rag_ready: bool
    model_id: str
    granite_status: str
