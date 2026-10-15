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

### 7. Ask a Question
```http
POST /video/ask
Content-Type: application/json
```

Answers a question about the **whole lecture** from its transcript. The current moment is optional context, not a restriction.

**Request:**
```json
{
  "video_id": "3zLQfuMZMF4",
  "question": "What did the instructor say about choosing u earlier?",
  "current_time_seconds": 187.5,
  "intervention_id": null
}
```

| Field | Required | Rules |
|---|---|---|
| `video_id` | yes | 11 characters from `A-Z a-z 0-9 _ -` |
| `question` | yes | trimmed, 1-500 characters |
| `current_time_seconds` | no | 0 or more; anchors phrases such as "the last five minutes" and "this" |
| `intervention_id` | no | must belong to `video_id`; adds that moment's explanation and reliable visual analysis; defaults the playback position to its timestamp |

**Response (200):**
```json
{
  "status": "answered",
  "scope": "lecture",
  "answer_text": "The instructor says to pick u as the function that comes first in the acronym LIATE...",
  "sources": [
    { "start_seconds": 60.0, "end_seconds": 100.0, "intervention_id": null }
  ]
}
```

- `status`: `answered`, `not_in_lecture`, or `insufficient_context`. Only `answered` has sources (at least one); the others have `sources: []` and a non-empty `answer_text` explaining why.
  - `not_in_lecture` is returned only when the model was given the **complete** transcript and judged the question unrelated. A search that finds nothing in a long transcript, or a model verdict reached from partial evidence, is `insufficient_context`: a keyword miss cannot prove the lecture lacks a topic.
- `scope`: `lecture`, or `lecture_with_moment` when a valid `intervention_id` was supplied.
- `sources`: time ranges computed by the backend, never produced by the model. The evidence is split into labelled passages (S1, S2, ...); the model may name the ones it relied on, and the backend keeps only ids it actually supplied (invented ids are ignored). With no valid citation, all supplied passages are reported. Adjacent passages are merged. A source with an `intervention_id` marks that moment's generated explanation/visual analysis.

**Errors** (`{"detail": "..."}`): `422` invalid request; `403` browser request from an origin that is not allowed (see Security below); `404` unknown video or intervention (including one belonging to another video); `429` provider quota or rate limit; `500` Q&A not configured (for example `GEMINI_API_KEY` missing), a database failure, or an unexpected error; `502` provider rejected credentials or the request, or returned an unusable reply (for example a malformed status marker); `503` provider unreachable or unavailable; `504` provider timeout. Messages never include keys, prompts, SQL, or raw provider errors.

**Model output contract:** the model may answer, or start its reply with exactly one `[[NOT_IN_LECTURE]]` or `[[INSUFFICIENT_CONTEXT]]` marker followed by at most one short sentence. Anything else that looks like a status marker (wrong case or brackets, a marker after other text, repeated or unknown markers, a marker followed by a long answer) is rejected as an unusable reply (`502`); it is never shown to the user and never turned into a `not_in_lecture` or `insufficient_context` result.

**Transcript source:** database rows are used when present; the JSON file is used only when the database has no usable rows for that video. Database errors are not hidden: they return `500`. A transcript file is used only if its folder name equals the video ID exactly, including case.

A video with a missing, empty, or malformed transcript returns `200` with `insufficient_context` (no LLM call is made).

**Configuration:** requires `GEMINI_API_KEY` (see `llm_interface/README.md`) when a request reaches the model. Ollama is not used. Token budget (all optional; token counts are local estimates):

| Variable | Default | Meaning |
|---|---|---|
| `ASK_MAX_PROMPT_TOKENS` | 4000 | Budget for the whole input: fixed instruction (about 650 tokens), question, moment context and evidence. What remains after the fixed parts becomes the evidence budget, so a long lecture cannot produce an oversized prompt. |
| `ASK_FULL_TRANSCRIPT_MAX_TOKENS` | 2000 | Transcripts up to this size are sent whole (about 7 minutes of speech in the sample lectures). |
| `ASK_CHARS_PER_TOKEN` | 3.5 | Characters per token used for the estimate. Deliberately below the usual ~4 for English because captions contain numbers and symbols. Calibrate it from the logged real counts. |
| `ASK_MAX_EVIDENCE_CHARS` | unset | Older knob, still honoured as an extra upper bound on evidence text. |

**How evidence is chosen:** transcript text for this video only (database rows first, then `data/<video_id>/transcript.json`).

1. A transcript within `ASK_FULL_TRANSCRIPT_MAX_TOKENS` is sent whole and marked complete. For such lectures retrieval cannot save meaningful tokens (retrieval still selected 62-82% of the sample lecture for most paraphrases, 33% for the narrowest), and a retrieval miss would be a false "not found".
2. An explicit time ("at 3:20", "between 2:00 and 4:00", "the last five minutes") sends only that stretch. "Summarize" sends the whole lecture or, if it does not fit, an even sample.
3. Otherwise BM25 ranking over ~40-second chunks, with: synonym and morphology groups (for example prioritize/decide/pick/select/choose, order/rank/priority, substitute/replace/plug, integrate/integral/integration), irregular stems (chosen/choose), single-letter symbols such as `u` and `x`, and the ASR sound alias `8` to `ate` (so `li8` matches `LIATE`). A gate on IDF-weighted coverage rejects questions whose only matches are ubiquitous words; chunks scoring under 35% of the best are dropped, and 25 seconds of context is added around each kept chunk so an explanation split across a chunk boundary stays whole.
4. If the gate fails on a long transcript the answer is `insufficient_context`, with no model call.

**Limitations of the retrieval:** matching is lexical. The synonym groups are a short hand-written list, not a thesaurus, so wording outside them (for example "speed" vs "velocity") can still miss. There is deliberately no fuzzy matching, because it matches common words. Ordinary words shared with the lecture can let an unrelated question through the gate (for example one that shares the rare-in-this-lecture word "first"); the model then receives a few chunks and should answer `insufficient_context`, at the cost of one request. Whole-lecture questions on transcripts above the budget see an even sample. Time phrases are English only.

**Token use:** each model call logs one line (`Ask tokens: ...`) with the local estimate next to the counts Gemini reports (`actual_prompt`, `output`, `thoughts`, `total`, `cached`). No question, answer or transcript text is logged. No separate `count_tokens` request is made, because that is an extra request against a free-tier quota that is usually limited by requests. Gemini context caching could cut the repeated cost of the fixed instruction or of a whole short transcript, but it has a minimum cached size and storage/quota implications and adds a cache lifecycle, so it is left as a later optimisation once real usage numbers exist.

### Security: who can call the API

The server binds `127.0.0.1` (see `api/main.py` and the command below), so other machines cannot reach it. This is not authentication: any program on this computer, and any web page open in a browser on it, can still try to call it.

- CORS allows only the Synapse frontend origin(s). Default: `http://localhost:5173` and `http://127.0.0.1:5173` (the Vite dev server used by `npm run electron:dev`). Override with `SYNAPSE_CORS_ORIGINS`, a comma-separated list of origins such as `http://localhost:5173,https://app.example.com`. Wildcards and malformed entries are ignored. The variable is read when the server starts. Credentials (cookies) are not allowed.
- `POST /video/ask` additionally rejects a browser request whose `Origin` header is not allowed (`403`). Requests without an `Origin` header (curl, scripts, Electron's main process) are not blocked.
- A packaged Electron build loads the page from a `file://` URL, which sends `Origin: null`. `null` is not allowed by default, because sandboxed iframes on any web page also send `Origin: null`. Add `null` to `SYNAPSE_CORS_ORIGINS` only if you accept that.
- The other endpoints are limited by CORS only (no `Origin` enforcement).

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
