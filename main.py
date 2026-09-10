"""
StartupGenie – Main FastAPI Application Entry Point
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import get_settings
from rag.retriever import RAGRetriever
from granite.granite_client import GraniteClient
from granite.blueprint_generator import BlueprintGenerator
from backend.routes import router

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Application Lifespan
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise services on startup and clean up on shutdown."""
    logger.info("=== StartupGenie starting up ===")
    settings = get_settings()

    # Initialise RAG Retriever
    rag_retriever = RAGRetriever(
        kb_dir="data/knowledge_base",
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k_results,
    )
    rag_retriever.initialize()
    app.state.rag_retriever = rag_retriever

    # Initialise IBM Granite Client
    granite_client = GraniteClient(
        api_key=settings.watsonx_api_key,
        project_id=settings.watsonx_project_id,
        url=settings.watsonx_url,
        model_id=settings.granite_model_id,
    )

    # Initialise Blueprint Generator
    blueprint_generator = BlueprintGenerator(granite_client=granite_client)
    app.state.blueprint_generator = blueprint_generator

    logger.info("=== StartupGenie ready to serve requests ===")
    yield

    logger.info("=== StartupGenie shutting down ===")


# ──────────────────────────────────────────────
# Application Factory
# ──────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="StartupGenie – AI Startup Blueprint Generator",
        description=(
            "Transform your startup idea into a structured, actionable blueprint "
            "powered by IBM Granite LLM and Retrieval-Augmented Generation (RAG)."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS – allow all origins for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(router, prefix="/api/v1", tags=["blueprint"])

    # Serve frontend static files
    frontend_path = Path("frontend")
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            return FileResponse("frontend/index.html")

    return app


app = create_app()


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info",
    )
