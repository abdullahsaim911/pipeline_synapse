"""
API routers package.
"""

from .video import router as video_router
from .learning_history import router as learning_history_router

__all__ = ["video_router", "learning_history_router"]
