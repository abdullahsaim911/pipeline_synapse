import { useState, useEffect, useRef } from "react";
import jsPDF from "jspdf";
import Sidebar from "./Sidebar";
import {
  getExplanation,
  getInterventionPoints,
  getTTSExplanation,
} from "../api";
import {
  saveBookmark,
  removeBookmark,
  isBookmarked,
  saveToLibrary,
} from "../storage";

/**
 * Explanation — opens when the user clicks a point on the Timeline.
 *
 * Props:
 *   - jobId, videoId, pointId, mode
 *   - onBack(), onNavigate(key)
 */
export default function Explanation({
  jobId,
  videoId,
  pointId,
  mode = "standard",
  bookmarkData = null, // pre-populated when navigating from Bookmarks
  autoPlay = false, // auto-start audio when navigating from Library
  onBack = () => {},
  onNavigate = () => {},
}) {
  const initialPoint = bookmarkData
    ? {
        id: bookmarkData.pointId,
        title: bookmarkData.title,
        category: bookmarkData.category,
        timestamp: bookmarkData.timestamp,
      }
    : null;
  const preloadedExpl = useRef(bookmarkData?.explanation || null);

  const [allPoints, setAllPoints] = useState(
    initialPoint ? [initialPoint] : [],
  );
  const [videoTitle, setVideoTitle] = useState(bookmarkData?.videoTitle || "");
  const [activePointId, setActivePointId] = useState(pointId);
  const [explanation, setExplanation] = useState(
    bookmarkData?.explanation || null,
  );
  const [bookmarked, setBookmarked] = useState(false);

  // Audio playback — real HTML5 Audio via TTS backend
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);

  // Load all points (for the left sidebar contents list)
  useEffect(() => {
    getInterventionPoints(jobId, videoId)
      .then((result) => {
        setAllPoints(result.points);
        setVideoTitle(result.videoTitle || "");
      })
      .catch((err) => console.error("Failed to load points:", err));
  }, [jobId, videoId]);

  // Load text explanation + kick off TTS fetch when point/mode changes
  useEffect(() => {
    if (!preloadedExpl.current) setExplanation(null);
    setCurrentTime(0);
    setIsPlaying(false);
    setAudioUrl(null);
    setAudioLoading(true);

    setBookmarked(isBookmarked(`${videoId}-${activePointId}`));

    // Text explanation
    getExplanation(activePointId, mode, videoId)
      .then((data) => {
        setExplanation(data);
        preloadedExpl.current = null;
      })
      .catch((err) => console.error("Failed to load explanation:", err));

    // TTS audio — runs in parallel (shares the same HTTP request via cache)
    getTTSExplanation(activePointId, mode, videoId)
      .then((tts) => setAudioUrl(tts.audio_file_path))
      .catch((err) => console.error("TTS fetch failed:", err))
      .finally(() => setAudioLoading(false));
  }, [activePointId, mode, videoId]);

  // Wire up the HTML5 Audio element whenever the URL changes
  useEffect(() => {
    if (!audioUrl) return;

    const audio = new Audio(audioUrl);
    audioRef.current = audio;

    const onTimeUpdate = () => setCurrentTime(audio.currentTime);
    const onLoaded = () => setAudioDuration(audio.duration);
    const onEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("loadedmetadata", onLoaded);
    audio.addEventListener("ended", onEnded);

    // Auto-play when coming from the Library "Play audio" button
    if (autoPlay) {
      audio
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => {});
    }

    return () => {
      audio.pause();
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("loadedmetadata", onLoaded);
      audio.removeEventListener("ended", onEnded);
      audioRef.current = null;
    };
  }, [audioUrl]);

  // ---- Actions -------------------------------------------------------------

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      if (audioDuration > 0 && currentTime >= audioDuration)
        audio.currentTime = 0;
      audio
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => {});
    }
  };

  const handleBookmark = () => {
    const id = `${videoId}-${activePointId}`;
    if (bookmarked) {
      removeBookmark(id);
      setBookmarked(false);
    } else if (activePoint && explanation) {
      saveBookmark({
        id,
        pointId: activePointId,
        videoId,
        jobId,
        videoTitle,
        title: activePoint.title,
        category: activePoint.category,
        timestamp: activePoint.timestamp,
        mode,
        explanation: {
          subtitle: explanation.subtitle,
          paragraphs: explanation.paragraphs || [],
          keyInsight: explanation.keyInsight,
        },
        savedAt: Date.now(),
      });
      setBookmarked(true);
    }
  };

  const handleExportPDF = () => {
    if (!explanation) return;
    exportToPDF(explanation, activePoint);
  };

  const handleExportAudio = () => {
    if (!explanation) return;
    exportToAudio(explanation, activePoint);
  };

  // ---- Derived data --------------------------------------------------------

  // Use a fallback so the screen renders even if allPoints hasn't loaded yet
  const activePoint = allPoints.find((p) => p.id === activePointId) ?? {
    id: activePointId,
    title: "Explanation",
    category: "diagram",
    timestamp: "",
  };

  const frameUrl = activePoint?._frame_path
    ? `http://127.0.0.1:8000/${activePoint._frame_path.replace(/^\.\//, "")}`
    : null;

  // Save to library once both explanation AND point metadata are ready.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!explanation || !activePoint) return;
    saveToLibrary({
      id: `${videoId}-${activePointId}`,
      pointId: activePointId,
      videoId,
      jobId,
      videoTitle,
      title: activePoint.title,
      category: activePoint.category,
      timestamp: activePoint.timestamp,
      mode,
      explanation,
      savedAt: Date.now(),
    });
  }, [explanation, activePoint?.id]); // stable: fires when explanation loads for a given point
  const pointsByCategory = {
    equation: allPoints.filter((p) => p.category === "equation"),
    diagram: allPoints.filter((p) => p.category === "diagram"),
    chart: allPoints.filter((p) => p.category === "chart"),
    graph: allPoints.filter((p) => p.category === "graph"),
  };

  const currentIndex = allPoints.findIndex((p) => p.id === activePointId);
  const prevPoint = currentIndex > 0 ? allPoints[currentIndex - 1] : null;
  const nextPoint =
    currentIndex < allPoints.length - 1 ? allPoints[currentIndex + 1] : null;

  if (!explanation) {
    return (
      <div className="flex h-full w-full bg-paper">
        <Sidebar active="home" onNavigate={onNavigate} />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="font-serif text-[18px] text-ink mb-3">
              Generating explanation…
            </div>
            <div className="font-sans text-[12px] text-ink-ghost mb-6">
              The AI is reading the frame and writing your explanation.
              <br />
              This can take 1–2 minutes.
            </div>
            <div className="flex gap-1.5 justify-center">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="w-1.5 h-1.5 rounded-full bg-accent"
                  style={{
                    animation: `pulse 1.2s ease-in-out ${i * 0.4}s infinite`,
                  }}
                />
              ))}
            </div>
            <style>{`@keyframes pulse { 0%,100%{opacity:0.2} 50%{opacity:1} }`}</style>
          </div>
        </main>
      </div>
    );
  }

  // ---- Render --------------------------------------------------------------

  return (
    <div className="flex h-full w-full bg-paper overflow-hidden">
      <Sidebar active="home" onNavigate={onNavigate} />

      {/* Inner contents sidebar — list of all points by category */}
      <div className="w-[220px] bg-paper-warm border-r border-paper-edge py-7 overflow-y-auto shrink-0">
        <div className="px-5 mb-5">
          <div className="eyebrow text-eyebrow-sm mb-1">Contents</div>
          <button
            onClick={onBack}
            className="font-serif text-[13px] text-ink-muted hover:text-ink"
          >
            ←{" "}
            {bookmarkData
              ? "Back to bookmarks"
              : autoPlay
                ? "Back to library"
                : "Back to timeline"}
          </button>
        </div>

        <CategorySection
          label="Equations"
          count={pointsByCategory.equation.length}
          points={pointsByCategory.equation}
          activeId={activePointId}
          onSelect={setActivePointId}
        />
        <CategorySection
          label="Diagrams"
          count={pointsByCategory.diagram.length}
          points={pointsByCategory.diagram}
          activeId={activePointId}
          onSelect={setActivePointId}
        />
        <CategorySection
          label="Charts"
          count={pointsByCategory.chart.length}
          points={pointsByCategory.chart}
          activeId={activePointId}
          onSelect={setActivePointId}
        />
        <CategorySection
          label="Graphs"
          count={pointsByCategory.graph.length}
          points={pointsByCategory.graph}
          activeId={activePointId}
          onSelect={setActivePointId}
        />
      </div>

      {/* Main reading area */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-[720px] px-14 py-10">
          {/* Eyebrow */}
          <div className="font-sans text-[10px] text-ink-ghost uppercase tracking-[0.12em] mb-4 flex items-center gap-2">
            <span className="capitalize">{activePoint.category}</span>
            <span className="text-paper-rule">·</span>
            <span>Timestamp {activePoint.timestamp}</span>
            <span className="text-paper-rule">·</span>
            <span className="capitalize">{mode} depth</span>
          </div>

          {/* Title + subtitle */}
          <h1 className="headline text-[34px]">{activePoint.title}</h1>
          <p className="subtitle text-[17px] mt-2.5">{explanation.subtitle}</p>

          {/* Action strip */}
          <div className="flex gap-2 py-5 mt-7 border-t border-b border-paper-edge">
            <button
              onClick={togglePlay}
              disabled={!audioUrl}
              className="btn-secondary disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <span>{isPlaying ? "❚❚" : "♪"}</span>
              {audioLoading ? "Loading…" : isPlaying ? "Pause" : "Listen"}
            </button>
            <button onClick={handleBookmark} className="btn-secondary">
              <span>{bookmarked ? "★" : "❏"}</span>
              {bookmarked ? "Bookmarked" : "Bookmark"}
            </button>
            <button onClick={handleExportPDF} className="btn-secondary">
              <span>↓</span>
              Export PDF
            </button>
            <button onClick={handleExportAudio} className="btn-secondary">
              <span>↓</span>
              Export audio
            </button>
          </div>

          {/* Figure */}
          <figure className="my-9">
            <div className="aspect-video rounded-editorial overflow-hidden relative bg-gradient-to-br from-[#4A5B8C] to-[#8B6B4A] shadow-[0_1px_4px_rgba(42,37,32,0.1)]">
              {frameUrl ? (
                <img
                  src={frameUrl}
                  alt="Frame from video"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                    const fallback = e.currentTarget.parentElement.querySelector(".fallback-img");
                    if (fallback) fallback.style.display = "block";
                  }}
                />
              ) : null}
              {videoId && (
                <img
                  src={`https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`}
                  alt="Thumbnail"
                  className="fallback-img w-full h-full object-cover"
                  style={{ display: frameUrl ? "none" : "block" }}
                  onError={(e) => (e.currentTarget.style.display = "none")}
                />
              )}
            </div>
            <figcaption className="font-serif text-[13px] text-ink-ghost mt-3 leading-snug">
              Figure. The frame at {activePoint.timestamp} where this{" "}
              {activePoint.category} appears.
            </figcaption>
          </figure>

          {/* Body prose */}
          <div className="font-serif text-[17px] leading-[1.75] text-ink-soft">
            {explanation.paragraphs.map((para, i) => (
              <p key={i} className="mb-[18px]">
                {i === 0 ? (
                  <>
                    <span className="font-serif text-[42px] leading-[0.9] float-left mr-2.5 mt-2 text-ink font-medium">
                      {para.charAt(0)}
                    </span>
                    {para.slice(1)}
                  </>
                ) : (
                  para
                )}
              </p>
            ))}
          </div>

          {/* Key insight pull quote */}
          {explanation.keyInsight && (
            <blockquote className="my-7 pl-6 py-1 border-l-2 border-accent text-ink-muted">
              <p className="m-0 text-[17px] leading-[1.6]">
                {explanation.keyInsight}
              </p>
              <p className="font-sans text-[11px] text-ink-ghost mt-2.5 uppercase tracking-[0.08em]">
                Key insight
              </p>
            </blockquote>
          )}

          {/* Audio player strip */}
          <div className="mt-9 px-5 py-4 bg-paper-soft border border-paper-edge rounded-editorial flex items-center gap-4">
            {/* Play / Pause button */}
            <button
              onClick={togglePlay}
              disabled={!audioUrl}
              className="w-9 h-9 rounded-full bg-ink text-paper flex items-center justify-center shrink-0 hover:bg-[#0f0c08] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {audioLoading ? (
                <span className="text-[8px]">…</span>
              ) : (
                <span className="text-xs ml-0.5">{isPlaying ? "❚❚" : "▶"}</span>
              )}
            </button>

            <div className="flex-1">
              {/* Progress track */}
              <div className="h-0.5 bg-paper-edge rounded-sm relative">
                <div
                  className="h-full bg-ink rounded-sm transition-all duration-100"
                  style={{
                    width:
                      audioDuration > 0
                        ? `${(currentTime / audioDuration) * 100}%`
                        : "0%",
                  }}
                />
                {audioDuration > 0 && (
                  <div
                    className="absolute top-1/2 w-2.5 h-2.5 bg-ink rounded-full transition-all duration-100"
                    style={{
                      left: `${(currentTime / audioDuration) * 100}%`,
                      transform: "translate(-50%, -50%)",
                    }}
                  />
                )}
              </div>

              {/* Time row */}
              <div className="flex justify-between mt-2 font-mono text-[10px] text-ink-ghost">
                <span>{formatTime(currentTime)}</span>
                <span className="font-sans text-ink-faded">
                  {audioLoading ? "Loading audio…" : "Aria · 1.0×"}
                </span>
                <span>
                  {audioDuration > 0 ? formatTime(audioDuration) : "--:--"}
                </span>
              </div>
            </div>
          </div>

          {/* Prev/next navigation */}
          <div className="flex justify-between mt-9 pt-6 border-t border-paper-edge font-sans text-[12px]">
            {prevPoint ? (
              <button
                onClick={() => setActivePointId(prevPoint.id)}
                className="text-left hover:text-ink transition-colors"
              >
                <div className="text-[10px] text-ink-ghost uppercase tracking-[0.12em] mb-1">
                  ← Previous
                </div>
                <div className="font-serif text-[14px] text-ink-muted">
                  {prevPoint.title}
                </div>
              </button>
            ) : (
              <div />
            )}

            {nextPoint && (
              <button
                onClick={() => setActivePointId(nextPoint.id)}
                className="text-right hover:text-ink transition-colors"
              >
                <div className="text-[10px] text-ink-ghost uppercase tracking-[0.12em] mb-1">
                  Next →
                </div>
                <div className="font-serif text-[14px] text-ink-muted">
                  {nextPoint.title}
                </div>
              </button>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

/* ============================================================================
   Sub-components
   ============================================================================ */

function CategorySection({ label, count, points, activeId, onSelect }) {
  return (
    <div className="mb-5">
      <div className="px-5 py-1.5 flex justify-between font-sans text-[10px] text-ink-ghost uppercase tracking-[0.14em]">
        <span>{label}</span>
        <span>{count}</span>
      </div>
      {points.map((point) => {
        const isActive = point.id === activeId;
        return (
          <button
            key={point.id}
            onClick={() => onSelect(point.id)}
            className={
              "w-full text-left px-5 py-2 flex justify-between items-center transition-colors " +
              (isActive
                ? "bg-paper-soft border-l-[3px] border-ink"
                : "hover:bg-paper-soft/50 border-l-[3px] border-transparent")
            }
          >
            <span
              className={
                "font-sans text-[12px] " +
                (isActive ? "text-ink font-medium" : "text-ink-muted")
              }
            >
              {point.title}
            </span>
            <span
              className={
                "font-mono text-[11px] " +
                (isActive ? "text-ink" : "text-ink-ghost")
              }
            >
              {point.timestamp}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ============================================================================
   Helpers — exports + formatting
   ============================================================================ */

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * Generates a real PDF download using jsPDF.
 */
function exportToPDF(explanation, point) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 60;
  const contentWidth = pageWidth - margin * 2;
  let y = margin;

  // Eyebrow
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(138, 128, 112);
  doc.text(`${point.category.toUpperCase()} · ${point.timestamp}`, margin, y);
  y += 25;

  // Title
  doc.setFont("times", "normal");
  doc.setFontSize(24);
  doc.setTextColor(30, 26, 21);
  const titleLines = doc.splitTextToSize(point.title, contentWidth);
  doc.text(titleLines, margin, y);
  y += titleLines.length * 28 + 6;

  // Subtitle
  doc.setFont("times", "italic");
  doc.setFontSize(13);
  doc.setTextColor(92, 82, 71);
  const subLines = doc.splitTextToSize(explanation.subtitle, contentWidth);
  doc.text(subLines, margin, y);
  y += subLines.length * 16 + 20;

  // Divider
  doc.setDrawColor(229, 223, 208);
  doc.line(margin, y, pageWidth - margin, y);
  y += 24;

  // Body paragraphs
  doc.setFont("times", "normal");
  doc.setFontSize(12);
  doc.setTextColor(42, 37, 32);

  explanation.paragraphs.forEach((para) => {
    const lines = doc.splitTextToSize(para, contentWidth);
    lines.forEach((line) => {
      if (y > 770) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin, y);
      y += 18;
    });
    y += 8; // paragraph spacing
  });

  // Key insight
  if (explanation.keyInsight) {
    if (y > 720) {
      doc.addPage();
      y = margin;
    }
    y += 14;
    doc.setDrawColor(74, 91, 140);
    doc.setLineWidth(2);
    doc.line(margin, y - 6, margin, y + 40);
    doc.setLineWidth(1);

    doc.setFont("times", "italic");
    doc.setFontSize(12);
    doc.setTextColor(61, 53, 40);
    const insightLines = doc.splitTextToSize(
      explanation.keyInsight,
      contentWidth - 14,
    );
    doc.text(insightLines, margin + 14, y);
    y += insightLines.length * 16 + 8;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(138, 128, 112);
    doc.text("KEY INSIGHT", margin + 14, y);
  }

  // Footer
  doc.setFont("helvetica", "italic");
  doc.setFontSize(9);
  doc.setTextColor(138, 128, 112);
  doc.text(
    `Generated by Synapse — ${new Date().toLocaleDateString()}`,
    margin,
    810,
  );

  const filename = `synapse-${slugify(point.title)}.pdf`;
  doc.save(filename);
}

/**
 * Generates a real WAV audio file (silent placeholder for now — replace with
 * actual TTS audio when the backend is ready).
 *
 * Creates a 5-second silent WAV the user can save. Once your TTS pipeline is
 * wired up, fetch the audio file URL from the backend and download that
 * instead.
 */
function exportToAudio(explanation, point) {
  const sampleRate = 22050;
  const duration = 5; // seconds
  const numSamples = sampleRate * duration;

  // Create a low-volume tone as a placeholder so the file isn't completely
  // silent — students will know it played.
  const samples = new Int16Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    samples[i] = Math.sin((i / sampleRate) * 220 * 2 * Math.PI) * 1000;
    // Fade in/out
    if (i < sampleRate * 0.1) samples[i] *= i / (sampleRate * 0.1);
    if (i > numSamples - sampleRate * 0.1)
      samples[i] *= (numSamples - i) / (sampleRate * 0.1);
  }

  const wavBlob = createWavBlob(samples, sampleRate);
  const url = URL.createObjectURL(wavBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `synapse-${slugify(point.title)}.wav`;
  link.click();
  URL.revokeObjectURL(url);
}

/**
 * Builds a valid WAV file blob from raw 16-bit PCM samples.
 */
function createWavBlob(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");

  // fmt chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);

  // data chunk
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  // Sample data
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(44 + i * 2, samples[i], true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
}
