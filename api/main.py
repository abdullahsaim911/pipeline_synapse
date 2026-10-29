"""
Synapse API - FastAPI backend for the accessibility pipeline.

Serves the Synapse pipeline endpoints for video processing, intervention detection,
and explanation generation.
"""

from fastapi import FastAPI
import threading
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from api.database import SessionLocal
from api.explanation_preparation import ExplanationPreparationManager
from api.processing_jobs import ProcessingJobManager
from api.routers import learning_history_router, video_router
from api.routers import video as video_module
from api.routers.quiz import router as quiz_router
from api.logging_setup import configure_ask_logging
from api.security import configure_cors
from voice.routes import router as voice_router, warm_up as voice_warm_up

load_dotenv()
configure_ask_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background services after Alembic has provisioned the schema."""
    preparation = ExplanationPreparationManager(
        session_factory=SessionLocal,
        orchestrator_factory=video_module.get_orchestrator,
        data_dir=video_module.get_data_dir(),
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
    # Load the voice models in the background so the first spoken command is not slow.
    threading.Thread(target=voice_warm_up, daemon=True, name="voice-warmup").start()
    try:
        yield
    finally:
        processing.shutdown()
        preparation.shutdown()


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
app.include_router(voice_router)


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
