"""
Video router for Synapse API.

Handles video processing, metadata retrieval, and intervention management.
"""

import json
import logging
import os
import re
import sys
from typing import List
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path as ApiPath,
    Query,
    Request,
    Response,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, sessionmaker

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.database import get_db
from api.explanation_service import (
    ExplanationTargetNotFoundError,
    find_valid_cached_explanation,
    generate_or_get_explanation,
    resolve_explanation_audio_path,
)
from api.explanation_preparation import (
    PreparationNotRetryableError,
    PreparationRetryExhaustedError,
)
from api.models import Video, Intervention, Transcript, Explanation
from api.processing_jobs import (
    ProcessingJobManager,
    ProcessingJobNotFoundError,
    ProcessingJobTimeoutError,
)
from api.schemas import (
    ProcessVideoRequest,
    ProcessVideoResponse,
    VideoMetaResponse,
    InterventionResponse,
    BookmarkRequest,
    BookmarkResponse,
    CategoryCounts,
    GenerateExplanationRequest,
    GenerateExplanationResponse,
    AskRequest,
    AskResponse,
    AskSource,
    ExplanationPreparationItem,
    ExplanationPreparationStatusResponse,
    ExplanationRetryResponse,
    ProcessingJobResponse,
    VideoSearchResult,
)
from api.youtube_search import (
    YouTubeSearchService,
    YouTubeSearchNotConfiguredError,
    YouTubeSearchTimeoutError,
    YouTubeSearchRateLimitError,
    YouTubeSearchUnavailableError,
    YouTubeSearchProviderError,
    get_youtube_search_service,
)
from llm_interface.providers import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from api.security import enforce_allowed_origin
from qa.moment import InterventionNotFoundError, load_moment
from qa.faithfulness import build_gate_from_env
from qa.hybrid import build_selector_from_env
from qa.service import AskService
from qa.transcript import load_transcript

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/video", tags=["video"])

# Lazy load orchestrator to avoid PyTorch import issues on startup
orchestrator = None


def get_orchestrator():
    """Lazy load orchestrator when needed."""
    global orchestrator
    if orchestrator is None:
        from orchestrator.orchestrator import PipelineOrchestrator
        orchestrator = PipelineOrchestrator(base_dir="data")
    return orchestrator


def get_data_dir() -> str:
    """Root folder holding data/<video_id>/ (dependency, overridable in tests)."""
    return os.path.join(project_root, "data")


def get_preparation_manager(request: Request):
    """Return the lifespan-owned worker, or None in router-only test apps."""
    return getattr(request.app.state, "explanation_preparation", None)


def _processing_manager(request: Request, db: Session, preparation_manager):
    """Return the lifespan manager, with a lazy fallback for router-only apps."""
    manager = getattr(request.app.state, "processing_jobs", None)
    if manager is None:
        manager = ProcessingJobManager(
            session_factory=sessionmaker(bind=db.get_bind(), expire_on_commit=False),
            orchestrator_factory=get_orchestrator,
            preparation_manager=preparation_manager,
        )
        manager.start()
        request.app.state.processing_jobs = manager
    return manager


def _schedule_background_preparation(manager, video_id: str, output_mode: str) -> None:
    """Schedule best-effort background work without invalidating saved results."""
    if manager is None:
        return
    try:
        manager.schedule_video(video_id, output_mode)
    except Exception:
        logger.exception(
            "Could not schedule background explanations for video %s in mode %s",
            video_id,
            output_mode,
        )


def _audio_url(intervention_id: str, output_mode: str, audio_file_path: str | None) -> str | None:
    """Return the API resource for a generated audio file without exposing its path."""
    if not audio_file_path:
        return None
    return (
        f"/video/intervention/{quote(intervention_id, safe='')}/audio"
        f"?output_mode={quote(output_mode, safe='')}"
    )


def calculate_category_counts(interventions: List) -> CategoryCounts:
    """
    Count interventions by content type category.

    Args:
        interventions: List of InterventionResponse or Intervention objects

    Returns:
        CategoryCounts with counts for each category
    """
    counts = CategoryCounts()

    for intervention in interventions:
        content_type = getattr(intervention, 'content_type', None)
        if not content_type:
            content_type = getattr(intervention, 'content_type', 'text')

        # Map content type to category
        content_lower = content_type.lower()

        if 'equation' in content_lower:
            counts.equation += 1
        elif 'graph' in content_lower:
            counts.graph += 1
        elif 'circuit' in content_lower:
            counts.circuit += 1
        elif 'diagram' in content_lower:
            counts.diagram += 1
        elif 'code' in content_lower:
            counts.code += 1
        elif 'handwritten' in content_lower:
            counts.handwritten_notes += 1
        elif 'biology' in content_lower:
            counts.biology += 1
        elif 'chemistry' in content_lower:
            counts.chemistry += 1
        elif 'physics' in content_lower:
            counts.physics += 1
        elif 'slide' in content_lower:
            counts.slide += 1
        elif 'table' in content_lower:
            counts.table += 1
        elif 'text' in content_lower:
            counts.text += 1
        else:
            counts.unknown += 1

    return counts




# ============================================================================
# Endpoint: Search YouTube Lectures
# ============================================================================

@router.get("/search", response_model=List[VideoSearchResult])
def search_videos(
    q: str = Query(..., min_length=1, max_length=200, description="YouTube lecture search query"),
    search_service: YouTubeSearchService = Depends(get_youtube_search_service),
):
    """
    Search YouTube for public embeddable lectures <= 600s in length.

    Queries official YouTube Data API v3 using server-side YOUTUBE_API_KEY.
    Returns normalized objects: video_id, title, channel_name, thumbnail_url,
    duration_seconds, duration_formatted, canonical_url.
    """
    clean_query = q.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Search query must not be empty.")
    if len(clean_query) > 200:
        raise HTTPException(status_code=400, detail="Search query exceeds maximum length of 200 characters.")

    try:
        return search_service.search(clean_query)
    except YouTubeSearchNotConfiguredError as exc:
        logger.warning("YouTube search service not configured: %s", exc)
        raise HTTPException(status_code=503, detail="YouTube search is not configured.")
    except YouTubeSearchTimeoutError as exc:
        logger.warning("YouTube search provider timeout")
        raise HTTPException(status_code=504, detail="YouTube search request timed out.")
    except YouTubeSearchRateLimitError as exc:
        logger.warning("YouTube search rate limit or quota exceeded")
        raise HTTPException(status_code=429, detail="YouTube search rate limit or quota exceeded.")
    except YouTubeSearchUnavailableError as exc:
        logger.warning("YouTube search provider unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="YouTube search provider is currently unavailable.")
    except YouTubeSearchProviderError as exc:
        logger.warning("YouTube search provider error: %s", exc)
        raise HTTPException(status_code=502, detail="YouTube search provider returned an error.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during YouTube search")
        raise HTTPException(status_code=500, detail="Search failed.")


# ============================================================================
# Endpoint 1: Get Video Metadata
# ============================================================================

@router.get("/meta/{video_id}", response_model=VideoMetaResponse)
async def get_video_metadata(
    video_id: str = ApiPath(..., pattern=r"^[A-Za-z0-9_-]{11}$"),
    db: Session = Depends(get_db),
    data_dir: str = Depends(get_data_dir),
):
    """
    Get video metadata and thumbnail.

    Checks if video exists in database and returns metadata.
    If not found, checks if the video directory exists on disk.
    """
    video_dir = os.path.join(data_dir, video_id)
    meta = {}
    meta_path = os.path.join(video_dir, "video_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                loaded_meta = json.load(handle)
            if isinstance(loaded_meta, dict):
                meta = loaded_meta
        except (OSError, ValueError):
            logger.warning("Could not read video metadata for %s", video_id, exc_info=True)

    channel_name = (
        meta.get("channel_name")
        or meta.get("channel")
        or meta.get("uploader")
        or meta.get("author")
    )
    canonical_thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

    # Check database first
    video = db.query(Video).filter(Video.id == video_id).first()

    if video:
        return VideoMetaResponse(
            video_id=video.id,
            title=video.title,
            channel_name=channel_name,
            duration_seconds=video.duration_seconds,
            duration_formatted=video.duration_formatted,
            thumbnail_url=video.thumbnail_url or meta.get("thumbnail_url") or canonical_thumbnail,
            exists=True,
            processed=True
        )

    # Check if video directory exists on disk
    if os.path.exists(video_dir):
        if meta:
            return VideoMetaResponse(
                video_id=video_id,
                title=meta.get("title"),
                channel_name=channel_name,
                duration_seconds=meta.get("duration_seconds"),
                duration_formatted=meta.get("duration_formatted"),
                thumbnail_url=meta.get("thumbnail_url") or canonical_thumbnail,
                exists=True,
                processed=False  # Not yet in DB
            )
        return VideoMetaResponse(
            video_id=video_id,
            thumbnail_url=canonical_thumbnail,
            exists=True,
            processed=False
        )

    # Video doesn't exist
    return VideoMetaResponse(
        video_id=video_id,
        exists=False,
        processed=False
    )


# ============================================================================
# Endpoint 2: Process Video (Detection-Only)
# ============================================================================

def _processed_video_response(db: Session, video_id: str) -> ProcessVideoResponse:
    db.expire_all()
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=500, detail="Processed video cache is unavailable.")
    interventions = (
        db.query(Intervention)
        .filter(Intervention.video_id == video_id)
        .order_by(Intervention.timestamp)
        .all()
    )
    return ProcessVideoResponse(
        video_id=video_id,
        title=video.title,
        duration_seconds=video.duration_seconds,
        duration_formatted=video.duration_formatted,
        total_keyframes=video.total_keyframes,
        total_interventions=len(interventions),
        interventions=[InterventionResponse.model_validate(item) for item in interventions],
        category_counts=calculate_category_counts(interventions),
    )


@router.post(
    "/processing-jobs",
    response_model=ProcessingJobResponse,
    status_code=202,
)
def create_or_join_processing_job(
    request_body: ProcessVideoRequest,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
    preparation_manager=Depends(get_preparation_manager),
):
    """Create work for a video, or join its existing durable job/cache."""
    try:
        manager = _processing_manager(request, db, preparation_manager)
        snapshot = manager.create_or_join(
            request_body.youtube_url, request_body.output_mode
        )
        if snapshot["status"] == "succeeded":
            response.status_code = 200
        return ProcessingJobResponse(**snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Could not create or join a processing job")
        raise HTTPException(status_code=500, detail="Processing job could not be created.")


@router.get(
    "/processing-jobs/{job_id}",
    response_model=ProcessingJobResponse,
)
def get_processing_job_status(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    preparation_manager=Depends(get_preparation_manager),
):
    """Return persisted status and the latest real pipeline stage."""
    try:
        manager = _processing_manager(request, db, preparation_manager)
        return ProcessingJobResponse(**manager.get(job_id))
    except (ProcessingJobNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Processing job not found.")


@router.post(
    "/processing-jobs/{job_id}/retry",
    response_model=ProcessingJobResponse,
    status_code=202,
)
def retry_processing_job(
    job_id: str,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
    preparation_manager=Depends(get_preparation_manager),
):
    """Retry failed work, recover a stale lease, or reuse a completed cache."""
    try:
        manager = _processing_manager(request, db, preparation_manager)
        snapshot = manager.retry(job_id)
        if snapshot["status"] == "succeeded":
            response.status_code = 200
        return ProcessingJobResponse(**snapshot)
    except (ProcessingJobNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Processing job not found.")


@router.post("/process", response_model=ProcessVideoResponse)
def process_video(
    request_body: ProcessVideoRequest,
    request: Request,
    db: Session = Depends(get_db),
    preparation_manager=Depends(get_preparation_manager),
):
    """
    Process video and detect intervention points.

    Runs M0 (Frame Extraction) + M0b (Transcription) + M1 (Synchronization).
    Stops before VLM, Synthesis, and TTS.

    Input: { "youtube_url": "https://www.youtube.com/watch?v=..." }
    Output: Video metadata + list of intervention points
    """
    temporary_manager = None
    try:
        if getattr(request.app.state, "processing_jobs", None) is None:
            temporary_manager = _processing_manager(
                request, db, preparation_manager
            )
        manager = _processing_manager(request, db, preparation_manager)
        created = manager.create_or_join(
            request_body.youtube_url, request_body.output_mode
        )
        snapshot = manager.wait(created["job_id"])
        if snapshot["status"] == "failed":
            logger.error(
                "Video processing job %s failed with code %s",
                snapshot["job_id"],
                snapshot["error_code"],
            )
            if snapshot["error_code"] == "persistence_failed":
                detail = "Processing results could not be saved."
            else:
                detail = snapshot["error_message"] or "Video processing failed."
            raise HTTPException(status_code=500, detail=detail)

        video_id = snapshot["video_id"]
        return _processed_video_response(db, video_id)
    except ProcessingJobTimeoutError:
        raise HTTPException(status_code=504, detail="Video processing is still running.")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Video processing failed")
        raise HTTPException(status_code=500, detail="Video processing failed.")
    finally:
        if temporary_manager is not None:
            temporary_manager.shutdown()
            if getattr(request.app.state, "processing_jobs", None) is temporary_manager:
                del request.app.state.processing_jobs


# ============================================================================
# Endpoint 3: Generate Explanation for Intervention
# ============================================================================

@router.post("/intervention/explain", response_model=GenerateExplanationResponse)
def generate_explanation(
    request: GenerateExplanationRequest,
    db: Session = Depends(get_db),
    data_dir: str = Depends(get_data_dir),
    preparation_manager=Depends(get_preparation_manager),
):
    """
    Generate explanation for a specific intervention point.

    Uses orchestrator.generate_intervention_explanation() which reuses
    existing pipeline components (VLM, Synthesizer, TTS) with existing prompts.

    Input:
    {
        "video_id": "UoIIwzHug9M",
        "intervention_id": "uuid",
        "output_mode": "explanatory"  // brief, explanatory, detailed (default: explanatory)
    }

    Output:
    {
        "intervention_id": "uuid",
        "text_explanation": "...",
        "audio_file_path": "data/UoIIwzHug9M/interventions/550e8400/explanation.mp3",
        "output_mode": "explanatory"
    }

    Note: vlm_snapshot is NOT returned - frontend only needs final text/audio.
    If background preparation already owns this key, the shared generation
    lock makes this request wait and then reuse that job's validated cache.
    """
    try:
        # Bind fresh sessions to the same database as the request dependency,
        # while never sharing the request-scoped Session with worker threads.
        isolated_sessions = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
        if preparation_manager is not None:
            result = preparation_manager.generate_manual(
                request.video_id,
                request.intervention_id,
                request.output_mode,
            )
        else:
            result = generate_or_get_explanation(
                session_factory=isolated_sessions,
                orchestrator_factory=get_orchestrator,
                data_dir=data_dir,
                video_id=request.video_id,
                intervention_id=request.intervention_id,
                output_mode=request.output_mode,
            )
        return GenerateExplanationResponse(
            intervention_id=result.intervention_id,
            text_explanation=result.text_explanation,
            audio_file_path=result.audio_file_path,
            audio_url=_audio_url(
                result.intervention_id,
                result.output_mode,
                result.audio_file_path,
            ),
            text_file_path=result.text_file_path,
            vlm_snapshot_path=result.vlm_snapshot_path,
            output_mode=result.output_mode,
        )
    except ExplanationTargetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PreparationRetryExhaustedError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "retry_exhausted",
                "attempt_count": exc.attempt_count,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception(
            "Explanation generation failed for intervention %s",
            request.intervention_id,
        )
        raise HTTPException(status_code=500, detail="Explanation generation failed.")


# ============================================================================
# Persistent Explanation Preparation Status and Retry
# ============================================================================

@router.get(
    "/{video_id}/explanations/status",
    response_model=ExplanationPreparationStatusResponse,
)
def get_explanation_preparation_status(
    video_id: str = ApiPath(..., pattern=r"^[A-Za-z0-9_-]{11}$"),
    output_mode: str = Query(
        default="explanatory",
        pattern=r"^(brief|explanatory|detailed)$",
    ),
    preparation_manager=Depends(get_preparation_manager),
):
    """Return compact, timestamp-ordered preparation state for one mode."""
    if preparation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Explanation preparation is unavailable.",
        )
    try:
        items = preparation_manager.get_status(video_id, output_mode)
        return ExplanationPreparationStatusResponse(
            video_id=video_id,
            output_mode=output_mode,
            items=[
                ExplanationPreparationItem(
                    intervention_id=item.intervention_id,
                    timestamp=item.timestamp,
                    status=item.status,
                    error_code=item.error_code,
                    attempt_count=item.attempt_count,
                )
                for item in items
            ],
        )
    except ExplanationTargetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post(
    "/{video_id}/interventions/{intervention_id}/explanations/retry",
    response_model=ExplanationRetryResponse,
    status_code=202,
    dependencies=[Depends(enforce_allowed_origin)],
)
def retry_explanation_preparation(
    response: Response,
    video_id: str = ApiPath(..., pattern=r"^[A-Za-z0-9_-]{11}$"),
    intervention_id: str = ApiPath(..., pattern=r"^[A-Za-z0-9-]{1,64}$"),
    output_mode: str = Query(
        default="explanatory",
        pattern=r"^(brief|explanatory|detailed)$",
    ),
    preparation_manager=Depends(get_preparation_manager),
):
    """Requeue one failed preparation key without waiting for inference."""
    if preparation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Explanation preparation is unavailable.",
        )
    try:
        result = preparation_manager.retry(
            video_id, intervention_id, output_mode
        )
        if not result.queued:
            response.status_code = 200
        return ExplanationRetryResponse(
            video_id=video_id,
            intervention_id=intervention_id,
            output_mode=output_mode,
            status=result.status,
            error_code=result.error_code,
            attempt_count=result.attempt_count,
        )
    except ExplanationTargetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PreparationNotRetryableError:
        raise HTTPException(
            status_code=409,
            detail={"code": "not_failed"},
        )
    except PreparationRetryExhaustedError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "retry_exhausted",
                "attempt_count": exc.attempt_count,
            },
        )


# ============================================================================
# Endpoint 4: Toggle Bookmark
# ============================================================================

@router.post("/intervention/bookmark/{intervention_id}", response_model=BookmarkResponse)
async def toggle_bookmark(
    intervention_id: str,
    request: BookmarkRequest,
    db: Session = Depends(get_db)
):
    """
    Toggle bookmark status for an intervention point.

    Input: { "bookmarked": true }
    Output: Updated bookmark status
    """
    intervention = db.query(Intervention).filter(
        Intervention.id == intervention_id
    ).first()

    if not intervention:
        raise HTTPException(status_code=404, detail="Intervention not found")

    intervention.is_bookmarked = request.bookmarked
    db.commit()
    db.refresh(intervention)

    return BookmarkResponse(
        intervention_id=intervention_id,
        is_bookmarked=intervention.is_bookmarked
    )


# ============================================================================
# Endpoint 5: Get Interventions
# ============================================================================

@router.get("/{video_id}/interventions", response_model=ProcessVideoResponse)
async def get_interventions(
    video_id: str,
    db: Session = Depends(get_db)
):
    """
    Get every persisted intervention point for a video.

    Returns interventions in educational playback order. Explanation records
    are intentionally not part of this response model and are not required.
    Structure matches Endpoint 2's ProcessVideoResponse.

    Args:
        video_id: YouTube video ID

    Returns:
        ProcessVideoResponse with video metadata and all interventions
    """
    # 1. Validate video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # 2. Get all interventions, including those not yet explained.
    interventions = db.query(Intervention).filter(
        Intervention.video_id == video_id
    ).order_by(
        Intervention.timestamp.asc(),
        Intervention.id.asc()
    ).all()

    # 3. Convert to response models
    intervention_responses = [InterventionResponse.model_validate(i) for i in interventions]

    # 4. Calculate category counts
    category_counts = calculate_category_counts(interventions)

    # 5. Return response matching Endpoint 2 structure
    return ProcessVideoResponse(
        video_id=video.id,
        title=video.title,
        duration_seconds=video.duration_seconds,
        duration_formatted=video.duration_formatted,
        total_keyframes=video.total_keyframes,
        total_interventions=len(intervention_responses),
        interventions=intervention_responses,
        category_counts=category_counts
    )


# ============================================================================
# Endpoint 6: Get Explanation by Intervention ID
# ============================================================================

@router.get("/intervention/{intervention_id}/explanation", response_model=GenerateExplanationResponse)
async def get_intervention_explanation(
    intervention_id: str,
    output_mode: str = "explanatory",
    db: Session = Depends(get_db)
):
    """
    Get explanation for a specific intervention if already generated.

    Returns the same structure as the explain endpoint.
    If explanation doesn't exist, returns 404.

    Args:
        intervention_id: UUID of the intervention
        output_mode: "brief", "explanatory" (default), or "detailed"

    Returns:
        GenerateExplanationResponse with text, audio, and file paths
    """
    # 1. Check if intervention exists
    intervention = db.query(Intervention).filter(
        Intervention.id == intervention_id
    ).first()

    if not intervention:
        raise HTTPException(status_code=404, detail="Intervention not found")

    # 2. Check if explanation exists for this intervention and mode
    explanation = db.query(Explanation).filter(
        Explanation.intervention_id == intervention_id,
        Explanation.output_mode == output_mode
    ).first()

    if not explanation:
        raise HTTPException(
            status_code=404,
            detail=f"Explanation not found for mode '{output_mode}'. Generate it first."
        )

    # 3. Return response matching Endpoint 3 structure
    return GenerateExplanationResponse(
        intervention_id=intervention_id,
        text_explanation=explanation.text_explanation,
        audio_file_path=explanation.audio_file_path,
        audio_url=_audio_url(
            intervention_id,
            explanation.output_mode,
            explanation.audio_file_path,
        ),
        text_file_path=explanation.text_file_path,
        vlm_snapshot_path=explanation.vlm_snapshot_path,
        output_mode=explanation.output_mode
    )
# ============================================================================
# Endpoint 7: Ask a Question (transcript-grounded, lecture-wide)
# ============================================================================

_ask_service = None


def get_ask_service() -> AskService:
    """Shared AskService (dependency, so tests can substitute a fake provider)."""
    global _ask_service
    if _ask_service is None:
        # build_selector_from_env() is None unless ASK_HYBRID_RETRIEVAL=1, so nothing heavy loads by default.
        _ask_service = AskService(evidence_selector=build_selector_from_env())
    return _ask_service


_answer_gate = None
_answer_gate_built = False


def get_answer_gate():
    """Faithfulness guardrail applied to answered responses (None while ASK_FAITHFULNESS_MODE is off). Built once; overridable in tests."""
    global _answer_gate, _answer_gate_built
    if not _answer_gate_built:
        _answer_gate = build_gate_from_env()
        _answer_gate_built = True
    return _answer_gate


@router.get("/intervention/{intervention_id}/audio", name="get_explanation_audio")
async def get_explanation_audio(
    intervention_id: str,
    output_mode: str = Query(
        default="explanatory",
        pattern=r"^(brief|explanatory|detailed)$",
    ),
    db: Session = Depends(get_db),
    data_dir: str = Depends(get_data_dir),
):
    """Serve one generated explanation MP3 referenced by the explanation database."""
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", intervention_id):
        raise HTTPException(status_code=422, detail="Invalid intervention identifier")

    explanation = find_valid_cached_explanation(
        db, intervention_id, output_mode, data_dir
    )
    if explanation is None:
        raise HTTPException(status_code=404, detail="Explanation audio not found")

    audio_path = resolve_explanation_audio_path(
        data_dir, explanation.audio_file_path
    )
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Explanation audio not found")

    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=f"{intervention_id}-{output_mode}.mp3",
    )


def _llm_http_error(exc: LLMProviderError) -> HTTPException:
    """Map provider failures to HTTP errors without exposing provider details."""
    if isinstance(exc, LLMConfigurationError):
        return HTTPException(status_code=500, detail="Question answering is not configured on this server.")
    if isinstance(exc, LLMRateLimitError):
        return HTTPException(status_code=429, detail="The AI service is busy or over its quota. Please try again shortly.")
    if isinstance(exc, LLMTimeoutError):
        return HTTPException(status_code=504, detail="The AI service took too long to respond. Please try again.")
    if isinstance(exc, (LLMConnectionError, LLMUnavailableError)):
        return HTTPException(status_code=503, detail="The AI service is temporarily unavailable. Please try again.")
    if isinstance(exc, LLMMalformedResponseError):
        return HTTPException(status_code=502, detail="The AI service returned an unusable response. Please try again.")
    if isinstance(exc, LLMAuthenticationError):
        return HTTPException(status_code=502, detail="The AI service rejected this server's credentials.")
    return HTTPException(status_code=502, detail="The AI service could not produce an answer.")


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(enforce_allowed_origin)])
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db),
    service: AskService = Depends(get_ask_service),
    data_dir: str = Depends(get_data_dir),
    gate=Depends(get_answer_gate)
):
    """
    Answer a question about the whole lecture using its transcript.

    The current moment is optional context, not a restriction: intervention_id
    adds that moment's explanation and reliable visual analysis, and
    current_time_seconds anchors phrases such as "the last five minutes".

    Input:
    {
        "video_id": "3zLQfuMZMF4",
        "question": "What did the instructor say about choosing u?",
        "current_time_seconds": 187.5,   // optional
        "intervention_id": null           // optional
    }

    Output: status (answered | not_in_lecture | insufficient_context), scope
    (lecture | lecture_with_moment), answer_text, and sources (backend-computed
    time ranges of the lecture material given to the model).

    Declared as a plain `def` so FastAPI runs it in a worker thread: the
    database, file and LLM calls block and must not stall the event loop that
    serves the other endpoints.
    """
    try:
        transcript = load_transcript(request.video_id, db, data_dir)

        video = db.query(Video).filter(Video.id == request.video_id).first()
        if video is None and not transcript.file_exists:
            raise HTTPException(status_code=404, detail="Video not found")

        moment = None
        if request.intervention_id:
            try:
                moment = load_moment(
                    db, request.video_id, request.intervention_id,
                    transcript.segments, data_dir
                )
            except InterventionNotFoundError:
                raise HTTPException(status_code=404, detail="Intervention not found")

        outcome = service.answer(
            question=request.question,
            segments=transcript.segments,
            current_time=request.current_time_seconds,
            moment=moment
        )
        if gate is not None:
            # Programmatic grounding check; may turn a low-faithfulness answer into insufficient_context.
            outcome = gate(outcome)

        return AskResponse(
            status=outcome.status,
            scope=outcome.scope,
            answer_text=outcome.answer_text,
            sources=[
                AskSource(
                    start_seconds=s.start_seconds,
                    end_seconds=s.end_seconds,
                    intervention_id=s.intervention_id
                )
                for s in outcome.sources
            ]
        )

    except HTTPException:
        raise
    except LLMProviderError as e:
        logger.warning("Ask failed for video %s: %s (upstream status %s)", request.video_id, e.code, e.status_code)
        raise _llm_http_error(e)
    except Exception:
        logger.exception("Unexpected error answering question for video %s", request.video_id)
        raise HTTPException(status_code=500, detail="Could not answer the question.")
