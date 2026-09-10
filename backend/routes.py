"""
Backend – API Routes
Defines all FastAPI route handlers.
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.models import (
    StartupInput,
    BlueprintResponse,
    HealthResponse,
    RetrievedContext,
    ReadinessScore,
    LabelCounts,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_services(request: Request):
    """Extract shared services from app state."""
    return request.app.state.blueprint_generator, request.app.state.rag_retriever


@router.post(
    "/generate",
    response_model=BlueprintResponse,
    summary="Generate a Startup Blueprint",
    description=(
        "Takes startup input, retrieves relevant knowledge from the RAG knowledge base, "
        "and generates a complete startup blueprint using IBM Granite LLM."
    ),
)
async def generate_blueprint(startup_input: StartupInput, request: Request):
    """Generate a complete startup blueprint using RAG + IBM Granite."""
    blueprint_gen, rag_retriever = _get_services(request)

    try:
        logger.info("Generating blueprint for idea: %s", startup_input.idea[:80])
        result = blueprint_gen.generate(
            startup_input=startup_input.model_dump(),
            rag_retriever=rag_retriever,
        )

        retrieved = [
            RetrievedContext(
                content=doc["content"],
                source=doc["source"],
                score=doc["score"],
            )
            for doc in result["retrieved_context"]
        ]

        score_data = result.get("readiness_score", {})
        readiness = ReadinessScore(
            idea_clarity=score_data.get("idea_clarity", 0),
            market_opportunity=score_data.get("market_opportunity", 0),
            technical_feasibility=score_data.get("technical_feasibility", 0),
            business_model_strength=score_data.get("business_model_strength", 0),
            team_resource_readiness=score_data.get("team_resource_readiness", 0),
            regulatory_readiness=score_data.get("regulatory_readiness", 0),
            overall=score_data.get("overall", 0),
        )

        label_data = result.get("label_counts", {})
        labels = LabelCounts(
            retrieved_facts=label_data.get("retrieved_facts", 0),
            estimates=label_data.get("estimates", 0),
            assumptions=label_data.get("assumptions", 0),
            recommendations=label_data.get("recommendations", 0),
        )

        from config import get_settings
        settings = get_settings()

        return BlueprintResponse(
            startup_input=startup_input,
            blueprint_raw=result["blueprint_raw"],
            sections=result["sections"],
            retrieved_context=retrieved,
            readiness_score=readiness,
            label_counts=labels,
            model_used=settings.granite_model_id,
            rag_chunks_retrieved=len(retrieved),
        )

    except Exception as e:
        logger.exception("Blueprint generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Blueprint generation failed: {str(e)}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check that the RAG index and IBM Granite model are operational.",
)
async def health_check(request: Request):
    """Return the health status of the application."""
    blueprint_gen, rag_retriever = _get_services(request)

    rag_ready = rag_retriever._vector_store.is_ready
    granite_health = blueprint_gen.granite_client.health_check()

    from config import get_settings
    settings = get_settings()

    # "configured" means keys are present; "ok" (legacy) also counts as healthy
    granite_ok = granite_health["status"] in ("ok", "configured")

    return HealthResponse(
        status="ok" if rag_ready and granite_ok else "degraded",
        rag_ready=rag_ready,
        model_id=settings.granite_model_id,
        granite_status=granite_health["status"],
    )


@router.get(
    "/context",
    summary="Test RAG Retrieval",
    description="Retrieve relevant context from the knowledge base for a given query (for testing).",
)
async def get_context(query: str, top_k: int = 5, request: Request = None):
    """Retrieve RAG context for a free-text query."""
    _, rag_retriever = _get_services(request)
    if not rag_retriever._vector_store.is_ready:
        raise HTTPException(status_code=503, detail="RAG index not ready.")
    results = rag_retriever.retrieve(query, top_k=top_k)
    return {"query": query, "results": results, "count": len(results)}
