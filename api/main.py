"""
Synapse API - FastAPI backend for the accessibility pipeline.

Serves the Synapse pipeline endpoints for video processing, intervention detection,
and explanation generation.
"""

import sys
import time

from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from api.database import SessionLocal, engine
from api.explanation_preparation import ExplanationPreparationManager
from api.migrations import require_current_schema
from api.model_warmup import ModalWarmup
from api.processing_jobs import ProcessingJobManager
from api.routers import learning_history_router, video_router
from api.routers import video as video_module
from api.routers.quiz import router as quiz_router
from api.logging_setup import configure_ask_logging
from api.security import configure_cors
from stage_tracing import trace_stage

load_dotenv()
configure_ask_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background services after Alembic has provisioned the schema."""
    startup_started = time.monotonic()
    trace_stage("backend", "startup")
    require_current_schema(engine)
    # Warming runs in the background so an unavailable Modal never delays
    # startup. The VLM worker re-checks readiness before each extraction.
    warmup = ModalWarmup()
    warmup.start()
    app.state.model_warmup = warmup
    preparation = ExplanationPreparationManager(
        session_factory=SessionLocal,
        orchestrator_factory=video_module.get_orchestrator,
        data_dir=video_module.get_data_dir(),
        before_inference=warmup.ensure_ready,
    )
    preparation.start()
    app.state.explanation_preparation = preparation
    processing = ProcessingJobManager(
        session_factory=SessionLocal,
        orchestrator_factory=video_module.get_orchestrator,
        preparation_manager=preparation,
    )
    processing.start()
    app.state.processing_jobs = processing
    trace_stage("backend", "ready", started_at=startup_started)
    try:
        yield
    finally:
        shutdown_started = time.monotonic()
        trace_stage("backend", "shutdown_started")
        preparation.begin_shutdown()
        warmup.request_stop()
        processing.shutdown()
        preparation.shutdown()
        warmup.stop()
        # Loaded only if an explanation ran; stops only a server Synapse started.
        ollama = sys.modules.get("orchestrator.ollama_service")
        if ollama is not None:
            ollama.shutdown_shared_ollama()
        trace_stage("backend", "shutdown_complete", started_at=shutdown_started)


# Initialize FastAPI app
app = FastAPI(
    title="Synapse API",
    description="AI-powered accessibility pipeline for blind students",
    version="1.0.0",
    lifespan=lifespan
)

# CORS defaults to the Vite development origins. Packaged Electron's `null`
# origin is intentionally not enabled without a reviewed transport/auth design.
configure_cors(app)

# Include routers
app.include_router(video_router)
app.include_router(quiz_router)
app.include_router(learning_history_router)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "Synapse API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
