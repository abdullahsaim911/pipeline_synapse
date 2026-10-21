"""
Synapse API - FastAPI backend for the accessibility pipeline.

Serves the Synapse pipeline endpoints for video processing, intervention detection,
and explanation generation.
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from api.database import engine, Base
from api.routers import video_router
from api.routers.quiz import router as quiz_router
from api.logging_setup import configure_ask_logging
from api.security import configure_cors

load_dotenv()
configure_ask_logging()


# Create database tables on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup (if needed)
    pass


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
