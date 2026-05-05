# Synapse API - FastAPI Backend

FastAPI backend for the Synapse accessibility pipeline. Provides endpoints for video processing, intervention detection, and explanation generation.

## Project Structure

```
api/
├── main.py              # FastAPI application entry point
├── database.py          # SQLite database configuration
├── models.py            # SQLAlchemy database models
├── schemas.py           # Pydantic request/response schemas
├── requirements.txt     # Python dependencies
├── routers/
│   ├── __init__.py
│   └── video.py         # Video-related endpoints
└── synapse.db           # SQLite database (auto-created)
```

## Database Schema

### Tables

| Table | Description |
|-------|-------------|
| `videos` | Video metadata (title, duration, YouTube URL) |
| `interventions` | Detected intervention points (linked to keyframes) |
| `explanations` | Generated text/audio explanations |
| `transcripts` | Transcript segments for each video |

## API Endpoints

### 1. Get Video Metadata
```http
GET /video/meta/{video_id}
```

Returns video metadata and checks if video exists.

**Response:**
```json
{
  "video_id": "NFZtjSeT3XE",
  "title": "How To Draw Lewis Structures",
  "duration_seconds": 710,
  "duration_formatted": "11:50",
  "thumbnail_url": "...",
  "exists": true,
  "processed": true
}
```

### 2. Process Video (Detection-Only)
```http
POST /video/process
Content-Type: application/json
```

Process video and detect intervention points. Runs M0 (Frame Extraction) + M0b (Transcription) + M1 (Synchronization).

**Request:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=NFZtjSeT3XE"
}
```

**Response:**
```json
{
  "video_id": "NFZtjSeT3XE",
  "title": "How To Draw Lewis Structures",
  "duration_seconds": 710,
  "duration_formatted": "11:50",
  "total_keyframes": 17,
  "total_interventions": 5,
  "interventions": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "video_id": "NFZtjSeT3XE",
      "timestamp": 15.5,
      "timestamp_formatted": "00:15",
      "frame_path": "data/NFZtjSeT3XE/keyframes/frame_0001.jpg",
      "content_type": "chemistry",
      "complexity_score": 0.45,
      "trigger_reason": "Deictic + High Complexity",
      "confidence": "high",
      "is_bookmarked": false
    }
  ]
}
```

### 3. Generate Explanation
```http
POST /video/intervention/explain
Content-Type: application/json
```

Generate explanation (text + audio) for a specific intervention point. Runs VLM + Synthesizer + TTS.

**Request:**
```json
{
  "intervention_id": "550e8400-e29b-41d4-a716-446655440000",
  "output_mode": "explanatory"
}
```

**Response:**
```json
{
  "intervention_id": "550e8400-e29b-41d4-a716-446655440000",
  "text_explanation": "Look at this molecule, which shows...",
  "audio_file_path": "data/NFZtjSeT3XE/interventions/550e8400/explanation.mp3",
  "output_mode": "explanatory",
  "vlm_snapshot": {
    "content_type": "chemistry",
    "visual_analysis": { ... }
  }
}
```

### 4. Toggle Bookmark
```http
POST /video/intervention/bookmark/{intervention_id}
Content-Type: application/json
```

Toggle bookmark status for an intervention point.

**Request:**
```json
{
  "bookmarked": true
}
```

**Response:**
```json
{
  "intervention_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_bookmarked": true
}
```

## Installation

### 1. Install Dependencies

```bash
pip install -r api/requirements.txt
```

### 2. Start the API Server

```bash
# Option 1: Run from project root
python -m api.main

# Option 2: Run with uvicorn directly
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`

### 3. Access Interactive Documentation

*   **Swagger UI**: http://127.0.0.1:8000/docs
*   **ReDoc**: http://127.0.0.1:8000/redoc

## Prerequisites

Before running the API, ensure:

1.  **Ollama is running**:
    ```bash
    ollama serve
    ```
    Verify with: `ollama list` (should show `mistral`)

2.  **Qwen2-VL models are downloaded**:
    Models should be in `models/Qwen2-VL-7B-Instruct/` or `models/Qwen2-VL-2B/`

3.  **FFmpeg is available**: (Included in `frame-extraction/ffmpeg/bin/`)

## Usage Example

### Process a Video

```python
import requests

# Process video
response = requests.post(
    "http://127.0.0.1:8000/video/process",
    json={"youtube_url": "https://www.youtube.com/watch?v=NFZtjSeT3XE"}
)

result = response.json()
print(f"Video ID: {result['video_id']}")
print(f"Total Interventions: {result['total_interventions']}")

# Get first intervention ID
first_intervention = result['interventions'][0]
intervention_id = first_intervention['id']
```

### Generate Explanation

```python
# Generate explanation for an intervention
response = requests.post(
    "http://127.0.0.1:8000/video/intervention/explain",
    json={
        "intervention_id": intervention_id,
        "output_mode": "explanatory"
    }
)

explanation = response.json()
print(f"Text: {explanation['text_explanation']}")
print(f"Audio: {explanation['audio_file_path']}")
```

## File Storage

All files are stored in the `data/` directory:

```
data/
└── {video_id}/
    ├── keyframes/          # Extracted keyframe images (.jpg)
    ├── audio_segments/     # Generated audio files (.mp3)
    ├── transcript.json     # Original transcript (backup)
    ├── keyframes.json      # Keyframe metadata (backup)
    └── video_meta.json     # Video metadata (backup)
```

The database stores only:
*   File paths (strings)
*   Text content (transcripts, explanations)
*   Metadata (timestamps, scores, etc.)

## Configuration

### Database Location

Default: `./synapse.db` (in project root)

To change location, modify `api/database.py`:
```python
SQLALCHEMY_DATABASE_URL = "sqlite:///path/to/your/database.db"
```

### Base Directory for Data

Default: `data/` (relative to project root)

To change, modify the orchestrator initialization in `api/routers/video.py`:
```python
orchestrator = PipelineOrchestrator(base_dir="your_custom_dir")
```

## Troubleshooting

### "Invalid YouTube URL format"

Ensure the URL is a valid YouTube URL:
*   `https://www.youtube.com/watch?v=VIDEO_ID`
*   `https://youtu.be/VIDEO_ID`

### "Failed to load VLM model"

1.  Check that model files exist in `models/` directory
2.  Ensure sufficient GPU memory (5GB+ for 7B model)
3.  Check GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`

### "Ollama connection failed"

1.  Ensure Ollama is running: `ollama serve`
2.  Verify Mistral model is installed: `ollama pull mistral`
3.  Check connection: `curl http://localhost:11434/api/tags`

## Development

### Running with Auto-Reload

```bash
uvicorn api.main:app --reload
```

### Database Queries

Use SQLite CLI:
```bash
sqlite3 synapse.db

# Example queries
SELECT * FROM videos;
SELECT * FROM interventions WHERE video_id = 'NFZtjSeT3XE';
SELECT * FROM explanations WHERE intervention_id = '...';
```
