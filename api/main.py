"""
Synapse API - FastAPI backend for the accessibility pipeline.

Serves the Synapse pipeline endpoints for video processing, intervention detection,
and explanation generation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.database import engine, Base
from api.routers import video_router


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

# Configure CORS for Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For Electron, we allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video_router)


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
