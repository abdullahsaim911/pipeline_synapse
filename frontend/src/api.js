const USE_MOCK = false;

const API_BASE_URL = "http://127.0.0.1:8000";

let cachedResult = null;

export async function processVideo({ url, videoId, mode }) {
  if (USE_MOCK) {
    console.log("[MOCK] processVideo called with:", { url, videoId, mode });
    await delay(400);
    return { jobId: "mock-job-" + Math.random().toString(36).slice(2, 10) };
  }

  const response = await fetch(`${API_BASE_URL}/video/process`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      youtube_url: `https://www.youtube.com/watch?v=${videoId}`,
    }),
  });
  if (!response.ok) throw new Error(`Process failed: ${response.status}`);
  const data = await response.json();

  // Cache so Timeline can reuse without calling backend again.
  cachedResult = { videoId, data };
  return { jobId: videoId };
}

/* -------------------------------------------------------------------------- */
/* getJobStatus                                                               */
/* -------------------------------------------------------------------------- */
/**
 * Polls the status of a processing job.
 * Returns: { stage, progress, message, done }
 *   stage: "downloading" | "extracting" | "detecting" | "generating" | "done"
 *   progress: 0–100
 */
let mockJobStartTime = null;

export async function getJobStatus(jobId) {
  if (USE_MOCK) {
    if (!mockJobStartTime) mockJobStartTime = Date.now();
    const elapsed = (Date.now() - mockJobStartTime) / 1000; // seconds
    const totalDuration = 12; // pretend processing takes 12 seconds
    const progress = Math.min(100, Math.round((elapsed / totalDuration) * 100));

    let stage, message;
    if (progress < 20) {
      stage = "downloading";
      message = "Fetching video from YouTube";
    } else if (progress < 45) {
      stage = "extracting";
      message = "Extracting keyframes for analysis";
    } else if (progress < 80) {
      stage = "detecting";
      message = `Qwen is reading frame ${Math.floor(progress * 10)} of 872`;
    } else if (progress < 100) {
      stage = "generating";
      message = "Writing explanations for each point";
    } else {
      stage = "done";
      message = "Complete";
    }

    await delay(150);
    return { stage, progress, message, done: progress >= 100 };
  }

  // --- Real API call ---
  const response = await fetch(`${API_BASE_URL}/api/jobs/${jobId}`);
  if (!response.ok) throw new Error(`Status fetch failed: ${response.status}`);
  return response.json();
}

/* -------------------------------------------------------------------------- */
/* getInterventionPoints                                                      */
/* -------------------------------------------------------------------------- */
/**
 * Once a job is done, this returns the detected intervention points.
 * Used by the Timeline screen.
 */
export async function getInterventionPoints(jobId, videoId) {
  if (USE_MOCK) {
    await delay(300);
    let meta = { title: "Untitled video", author: "Unknown" };
    try {
      meta = await getVideoMetadata(videoId);
    } catch (err) {
      console.error("Metadata fetch failed:", err);
    }
    return {
      jobId,
      videoTitle: meta.title,
      author: meta.author,
      duration: "14:32",
      points: [
        {
          id: "p1",
          category: "diagram",
          timestamp: "0:45",
          title: "Circular motion",
        },
        {
          id: "p2",
          category: "equation",
          timestamp: "1:22",
          title: "Fourier integral",
        },
        {
          id: "p3",
          category: "diagram",
          timestamp: "2:30",
          title: "Signal mixing",
        },
        {
          id: "p4",
          category: "chart",
          timestamp: "3:18",
          title: "Amplitude spectrum",
        },
        {
          id: "p5",
          category: "equation",
          timestamp: "4:44",
          title: "Euler's formula",
        },
        {
          id: "p6",
          category: "diagram",
          timestamp: "5:56",
          title: "Frequency decomposition",
        },
      ],
    };
  }

  // Real backend
  // Real backend — read from cache populated by processVideo
  if (!cachedResult || cachedResult.videoId !== videoId) {
    throw new Error("No cached result. Process the video first.");
  }
  const raw = cachedResult.data;

  let meta = { title: raw.title, author: "Unknown" };
  try {
    meta = await getVideoMetadata(videoId);
  } catch (err) {
    console.error("Metadata fetch failed:", err);
  }

  let duration = raw.duration_formatted;
  if (raw.duration_seconds === 0 && raw.interventions.length > 0) {
    const lastSeconds =
      raw.interventions[raw.interventions.length - 1].timestamp;
    duration = secondsToMMSS(lastSeconds + 60);
  }

  const points = raw.interventions
    .filter((iv) => iv.content_type !== "non-educational")
    .map((iv) => ({
      id: iv.id,
      category: iv.content_type === "slide" ? "diagram" : iv.content_type,
      timestamp: iv.timestamp_formatted.replace(/^00:/, ""),
      title: titleFromContext(iv.transcript_context, iv.content_type),
      _frame_path: iv.frame_path,
      _transcript: iv.transcript_context,
    }));

  return {
    jobId,
    videoTitle: meta.title,
    author: meta.author,
    duration,
    points,
  };
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

// Convert local file path to full URL
export function getFullUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  // Use custom protocol for local files in the data directory
  if (path.startsWith("data") || path.startsWith("data\\")) {
    const url = window.synapseProtocol?.getDataUrl(path) || "";
    console.log("[api.js] getFullUrl:", path, "->", url);
    console.log("[api.js] synapseProtocol available:", !!window.synapseProtocol);
    return url;
  }
  return `http://127.0.0.1:8000/${path.replace(/^\.\//, "")}`;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Reset the mock timer (call when starting a new job).
export function resetMockTimer() {
  mockJobStartTime = null;
}
export async function getVideoMetadata(videoId) {
  const url = `https://noembed.com/embed?url=https://www.youtube.com/watch?v=${videoId}`;
  const response = await fetch(url);
  const data = await response.json();
  console.log("Raw oEmbed response:", data);
  return { title: data.title, author: data.author_name };
}

/**
 * Fetches the TTS audio and text explanation for an intervention point.
 * Response shape from backend:
 *   { intervention_id, text_explanation, audio_file_path, output_mode }
 *
 * Returns the same shape but with audio_file_path as a fully-qualified URL
 * ready to pass to new Audio().
 */

// Cache so parallel calls for the same point share one HTTP request.
const _explanationCache = new Map();

async function getInterventionExplanation(videoId, interventionId, mode, signal) {
  const key = `${videoId}::${interventionId}::${mode}`;
  if (_explanationCache.has(key)) return _explanationCache.get(key);

  const promise = fetch(`${API_BASE_URL}/video/intervention/explain`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      video_id: videoId,
      intervention_id: interventionId,
      output_mode: mode,
    }),
    signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const errBody = await res.text().catch(() => "(no body)");
        console.error(
          `[Synapse] /video/intervention/explain ${res.status}:`,
          errBody,
        );
        throw new Error(`Explain failed: ${res.status} — ${errBody}`);
      }
      const data = await res.json();
      let audioUrl = null;
      if (data.audio_file_path) {
        // Construct full URL from local path using custom protocol
        audioUrl = getFullUrl(data.audio_file_path);
        console.log("[api.js] Audio URL constructed:", data.audio_file_path, "->", audioUrl);
      }
      return { ...data, audio_file_path: audioUrl };
    })
    .catch((err) => {
      // Don't cache failures — allow retry on next call
      _explanationCache.delete(key);
      throw err;
    });

  _explanationCache.set(key, promise);
  return promise;
}

export async function getTTSExplanation(interventionId, mode, videoId, signal) {
  if (USE_MOCK) {
    await delay(600);
    return {
      intervention_id: interventionId,
      text_explanation: "Mock textual explanation for this intervention point.",
      audio_file_path: createMockAudioUrl(),
      output_mode: mode,
    };
  }

  return getInterventionExplanation(videoId, interventionId, mode, signal);
}

/* -------------------------------------------------------------------------- */
/* createMockAudioUrl (mock only)                                             */
/* -------------------------------------------------------------------------- */
function createMockAudioUrl() {
  const sampleRate = 22050;
  const duration = 8;
  const numSamples = sampleRate * duration;
  const samples = new Int16Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    // Simple 220 Hz tone with fade in/out
    samples[i] = Math.sin((i / sampleRate) * 220 * 2 * Math.PI) * 1200;
    if (i < sampleRate * 0.15) samples[i] *= i / (sampleRate * 0.15);
    if (i > numSamples - sampleRate * 0.15)
      samples[i] *= (numSamples - i) / (sampleRate * 0.15);
  }

  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const ws = (off, str) => {
    for (let i = 0; i < str.length; i++)
      view.setUint8(off + i, str.charCodeAt(i));
  };
  ws(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ws(8, "WAVE");
  ws(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ws(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++)
    view.setInt16(44 + i * 2, samples[i], true);

  const blob = new Blob([buffer], { type: "audio/wav" });
  return URL.createObjectURL(blob);
}

export async function getExplanation(pointId, mode, videoId, signal) {
  if (USE_MOCK) {
    await delay(300);
    return {
      pointId,
      subtitle:
        "A clear walk-through of what's happening on screen and why it matters.",
      paragraphs: [
        "This visual moment shows a key concept that often confuses students seeing it for the first time. The relationship between the elements on screen isn't arbitrary — each piece is chosen to make an abstract idea concrete and visible.",
        "The trick is recognizing what each part of the figure represents. Once you map the visual elements to the underlying ideas, the whole picture starts to make sense as a connected story rather than a collection of unfamiliar symbols.",
        "Notice how the structure here mirrors the formal definition you'd find in a textbook, but with a crucial difference: the visual lets you see why the definition has to be the way it is. That's the real value of working through diagrams instead of just memorizing equations.",
      ],
      keyInsight:
        "Most STEM concepts have a 'why' that's invisible in formal notation. Visuals make that why visible — which is why your textbook author drew this in the first place.",
    };
  }

  const data = await getInterventionExplanation(videoId, pointId, mode, signal);
  return parseTextExplanation(data.text_explanation, pointId);
}

// Splits the raw text_explanation string into the { subtitle, paragraphs, keyInsight }
// shape the Explanation component expects.
function parseTextExplanation(text, pointId) {
  if (!text) {
    return { pointId, subtitle: "", paragraphs: [], keyInsight: "" };
  }

  const rawParagraphs = text
    .split(/\n{2,}/)
    .map((p) => p.replace(/\n/g, " ").trim())
    .filter(Boolean);

  const subtitle =
    rawParagraphs.length > 0
      ? (rawParagraphs[0].split(/[.!?]/)[0] + ".").trim()
      : "";

  const body =
    rawParagraphs.length > 2 ? rawParagraphs.slice(0, -1) : rawParagraphs;

  const keyInsight =
    rawParagraphs.length > 2 ? rawParagraphs[rawParagraphs.length - 1] : "";

  return { pointId, subtitle, paragraphs: body, keyInsight };
}
function secondsToMMSS(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// Generates a short, sentence-cased title from the transcript snippet.
// Backend doesn't give us titles, so we make readable ones from context.
function titleFromContext(transcript, contentType) {
  if (!transcript)
    return contentType.charAt(0).toUpperCase() + contentType.slice(1);
  const cleaned = transcript.replace(/\s+/g, " ").trim();
  // Take first 6 words, capitalize first letter
  const words = cleaned.split(" ").slice(0, 6).join(" ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}
