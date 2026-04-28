"""
Video router for Synapse API.

Handles video processing, metadata retrieval, and intervention management.
"""

import json
import os
import sys
import uuid
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.database import get_db
from api.models import Video, Intervention, Transcript, Explanation
from api.schemas import (
    ProcessVideoRequest,
    ProcessVideoResponse,
    VideoMetaResponse,
    InterventionResponse,
    BookmarkRequest,
    BookmarkResponse,
    CategoryCounts,
    GenerateExplanationRequest,
    GenerateExplanationResponse
)


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
# Endpoint 1: Get Video Metadata
# ============================================================================

@router.get("/meta/{video_id}", response_model=VideoMetaResponse)
async def get_video_metadata(video_id: str, db: Session = Depends(get_db)):
    """
    Get video metadata and thumbnail.

    Checks if video exists in database and returns metadata.
    If not found, checks if the video directory exists on disk.
    """
    # Check database first
    video = db.query(Video).filter(Video.id == video_id).first()

    if video:
        return VideoMetaResponse(
            video_id=video.id,
            title=video.title,
            duration_seconds=video.duration_seconds,
            duration_formatted=video.duration_formatted,
            thumbnail_url=video.thumbnail_url,
            exists=True,
            processed=True
        )

    # Check if video directory exists on disk
    video_dir = f"data/{video_id}"
    if os.path.exists(video_dir):
        # Try to load metadata from file
        meta_path = os.path.join(video_dir, "video_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return VideoMetaResponse(
                video_id=meta.get("video_id", video_id),
                title=meta.get("title"),
                duration_seconds=meta.get("duration_seconds"),
                duration_formatted=meta.get("duration_formatted"),
                thumbnail_url=meta.get("thumbnail_url"),
                exists=True,
                processed=False  # Not yet in DB
            )
        return VideoMetaResponse(
            video_id=video_id,
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

def save_to_database_task(video_id: str, result: dict, video_dir: str):
    """
    Background task to save video data to database.

    Args:
        video_id: YouTube video ID
        result: Result from orchestrator.process_detection_only()
        video_dir: Video directory path
    """
    from api.database import SessionLocal

    db = SessionLocal()
    try:
        # Check if video already exists
        existing_video = db.query(Video).filter(Video.id == video_id).first()
        if existing_video:
            return  # Skip if already processed

        # Create video record
        video = Video(
            id=video_id,
            title=f"Video {video_id}",
            source_url=result.get("source_url", f"https://www.youtube.com/watch?v={video_id}"),
            duration_seconds=result.get("duration_seconds"),
            duration_formatted=result.get("duration_formatted"),
            total_keyframes=result.get("metadata", {}).get("total_keyframes", 0),
            subject=result.get("subject")
        )
        db.add(video)
        db.flush()  # Get video.id without committing

        # Save transcript segments
        transcript_data = result.get("transcript", {})
        if transcript_data and "entries" in transcript_data:
            for entry in transcript_data["entries"]:
                transcript = Transcript(
                    video_id=video_id,
                    start_time=entry.get("start", 0.0),
                    end_time=entry.get("start", 0.0) + entry.get("duration", 0.0),
                    text=entry.get("text", "")
                )
                db.add(transcript)

        # Save intervention points
        intervention_points = result.get("intervention_points", [])
        for point in intervention_points:
            intervention_id = str(uuid.uuid4())
            intervention = Intervention(
                id=intervention_id,
                video_id=video_id,
                timestamp=point.timestamp,
                timestamp_formatted=f"{int(point.timestamp // 60):02d}:{int(point.timestamp % 60):02d}",
                frame_path=point.frame_path,
                content_type=point.content_type,
                complexity_score=point.complexity_score,
                clip_confidence=point.clip_confidence,
                trigger_reason=point.trigger_reason,
                confidence=point.confidence,
                transcript_context=point.transcript_context
            )
            db.add(intervention)

        db.commit()
        print(f"[API] Saved video {video_id} with {len(intervention_points)} interventions to database")

    except Exception as e:
        db.rollback()
        print(f"[API] Error saving to database: {e}", exc_info=True)
    finally:
        db.close()


@router.post("/process", response_model=ProcessVideoResponse)
def process_video(
    request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Process video and detect intervention points.

    Runs M0 (Frame Extraction) + M0b (Transcription) + M1 (Synchronization).
    Stops before VLM, Synthesis, and TTS.

    Input: { "youtube_url": "https://www.youtube.com/watch?v=..." }
    Output: Video metadata + list of intervention points
    """
    try:
        # Get orchestrator instance
        orchestrator_instance = get_orchestrator()

        # Extract video ID
        video_id = orchestrator_instance._extract_video_id(request.youtube_url)

        # Check if already processed
        existing_video = db.query(Video).filter(Video.id == video_id).first()
        if existing_video:
            # Return existing interventions
            interventions = db.query(Intervention).filter(
                Intervention.video_id == video_id
            ).order_by(Intervention.timestamp).all()

            return ProcessVideoResponse(
                video_id=video_id,
                title=existing_video.title,
                duration_seconds=existing_video.duration_seconds,
                duration_formatted=existing_video.duration_formatted,
                total_keyframes=existing_video.total_keyframes,
                total_interventions=len(interventions),
                interventions=[InterventionResponse.model_validate(i) for i in interventions],
                category_counts=calculate_category_counts(interventions)
            )

        # Run detection-only pipeline
        result = orchestrator_instance.process_detection_only(request.youtube_url)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Convert intervention points to response models
        intervention_responses = []
        for point in result.get("intervention_points", []):
            # Generate temp ID (will be real UUID after DB save)
            temp_id = str(uuid.uuid4())
            intervention_responses.append(InterventionResponse(
                id=temp_id,
                video_id=video_id,
                timestamp=point.timestamp,
                timestamp_formatted=f"{int(point.timestamp // 60):02d}:{int(point.timestamp % 60):02d}",
                frame_path=point.frame_path,
                content_type=point.content_type,
                complexity_score=point.complexity_score,
                clip_confidence=point.clip_confidence,
                trigger_reason=point.trigger_reason,
                confidence=point.confidence,
                transcript_context=point.transcript_context,
                is_bookmarked=False,
                created_at=datetime.now()
            ))

        # Add background task to save to database
        background_tasks.add_task(save_to_database_task, video_id, result, result.get("video_dir", f"data/{video_id}"))

        return ProcessVideoResponse(
            video_id=video_id,
            title=result.get("title"),
            duration_seconds=result.get("duration_seconds"),
            duration_formatted=result.get("duration_formatted"),
            total_keyframes=result.get("metadata", {}).get("total_keyframes", 0),
            total_interventions=len(intervention_responses),
            interventions=intervention_responses,
            category_counts=calculate_category_counts(intervention_responses)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ============================================================================
# Endpoint 3: Generate Explanation for Intervention
# ============================================================================

@router.post("/intervention/explain", response_model=GenerateExplanationResponse)
async def generate_explanation(
    request: GenerateExplanationRequest,
    db: Session = Depends(get_db)
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
    """
    try:
        # 1. Validate video exists in database
        video = db.query(Video).filter(Video.id == request.video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # 2. Get intervention from database
        intervention = db.query(Intervention).filter(
            Intervention.id == request.intervention_id,
            Intervention.video_id == request.video_id
        ).first()

        if not intervention:
            raise HTTPException(status_code=404, detail="Intervention not found")

        # 3. Check if explanation already exists for this mode (cache)
        existing_explanation = db.query(Explanation).filter(
            Explanation.intervention_id == request.intervention_id,
            Explanation.output_mode == request.output_mode
        ).first()

        if existing_explanation:
            print(f"[API] Returning cached explanation for {request.intervention_id}")
            return GenerateExplanationResponse(
                intervention_id=intervention.id,
                text_explanation=existing_explanation.text_explanation,
                audio_file_path=existing_explanation.audio_file_path,
                text_file_path=existing_explanation.text_file_path,
                vlm_snapshot_path=existing_explanation.vlm_snapshot_path,
                output_mode=request.output_mode
            )

        # 4. Get all transcripts for context (from file system, not database)
        transcript_path = os.path.join(project_root, "data", request.video_id, "transcript.json")
        all_transcripts = []
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
                all_transcripts = transcript_data.get("entries", [])
        else:
            print(f"[API] Transcript file not found: {transcript_path}")

        # 5. Call orchestrator function to generate explanation
        orchestrator_instance = get_orchestrator()

        # Check clip confidence and set content_type accordingly
        content_type_to_use = intervention.content_type or "unknown"
        if intervention.clip_confidence is not None and intervention.clip_confidence < 0.40:
            content_type_to_use = "unknown"
            print(f"[API] Low clip confidence ({intervention.clip_confidence:.2f}) - using 'unknown' content type")

        result = orchestrator_instance.generate_intervention_explanation(
            video_id=request.video_id,
            intervention_id=intervention.id,
            intervention_timestamp=intervention.timestamp,
            frame_path=intervention.frame_path,
            content_type=content_type_to_use,
            trigger_reason=intervention.trigger_reason or "VLM Analysis",
            transcript_entries=all_transcripts,
            output_mode=request.output_mode
        )

        # 6. Save explanation to database for caching
        new_explanation = Explanation(
            intervention_id=intervention.id,
            text_explanation=result["text_explanation"],
            audio_file_path=result["audio_file_path"],
            text_file_path=result.get("text_file_path"),
            vlm_snapshot_path=result.get("vlm_snapshot_path"),
            output_mode=result["output_mode"]
        )
        db.add(new_explanation)
        db.commit()

        # 7. Return response
        return GenerateExplanationResponse(
            intervention_id=result["intervention_id"],
            text_explanation=result["text_explanation"],
            audio_file_path=result["audio_file_path"],
            text_file_path=result.get("text_file_path"),
            vlm_snapshot_path=result.get("vlm_snapshot_path"),
            output_mode=result["output_mode"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")


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
# Endpoint 5: Get Interventions with Explanations
# ============================================================================

@router.get("/{video_id}/interventions", response_model=ProcessVideoResponse)
async def get_interventions_with_explanations(
    video_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all intervention points that have explanations for a video.

    Returns interventions in reverse chronological order (recent to old).
    Structure matches Endpoint 2's ProcessVideoResponse.

    Args:
        video_id: YouTube video ID

    Returns:
        ProcessVideoResponse with video metadata and interventions with explanations
    """
    # 1. Validate video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # 2. Get interventions that have explanations (join with Explanation table)
    interventions_with_explanations = db.query(Intervention).join(
        Explanation,
        Intervention.id == Explanation.intervention_id
    ).filter(
        Intervention.video_id == video_id
    ).order_by(
        Intervention.created_at.desc()  # Recent to old
    ).distinct().all()

    # 3. Convert to response models
    intervention_responses = [InterventionResponse.model_validate(i) for i in interventions_with_explanations]

    # 4. Calculate category counts
    category_counts = calculate_category_counts(interventions_with_explanations)

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
        text_file_path=explanation.text_file_path,
        vlm_snapshot_path=explanation.vlm_snapshot_path,
        output_mode=explanation.output_mode
    )
