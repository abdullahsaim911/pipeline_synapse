import React, { useState, useMemo } from "react";
import Sidebar from "./Sidebar";

export default function Home({ onProcess = () => {}, onNavigate = () => {} }) {
  const [url, setUrl] = useState("");
  const [mode, setMode] = useState("standard");

  const videoId = useMemo(() => extractYouTubeId(url), [url]);
  const isValidUrl = Boolean(videoId);
  const thumbnailUrl = videoId
    ? `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`
    : null;

  const modes = [
    {
      key: "brief",
      label: "Novice",
      description:
        "Conceptual foundations focusing on metaphors and simplified logic structures.",
      icon: <LightbulbIcon />,
    },
    {
      key: "standard",
      label: "Intermediate",
      description:
        "Technical depth with moderate jargon. Bridges the gap between theory and application.",
      recommended: true,
      icon: <NeuralIcon />,
    },
    {
      key: "detailed",
      label: "Specialist",
      description:
        "Rigorous academic analysis with full terminology and mathematical depth.",
      icon: <CompassIcon />,
    },
  ];

  const handleSubmit = () => {
    if (!isValidUrl) return;
    onProcess({ url, videoId, mode });
  };

  return (
    <div className="flex h-full w-full bg-paper overflow-hidden">
      <Sidebar active="home" onNavigate={onNavigate} />

      <main className="flex-1 overflow-y-auto">
        <div className="w-full max-w-[640px] mx-auto px-10 py-14">

          {/* Header */}
          <div className="mb-10">
            <div className="eyebrow mb-4">New study session</div>
            <h1 className="headline text-[36px]">Paste a video to begin.</h1>
            <p className="subtitle text-[16px] mt-4">
              See the unseeable in STEM with AI-powered visual explanations.
            </p>
          </div>

          {/* URL Input */}
          <div className="mb-8">
            <label className="flex items-center gap-3.5 px-[18px] py-3.5 bg-paper-soft border border-paper-edge rounded-editorial focus-within:border-accent transition-colors">
              <span className="font-sans text-eyebrow pr-3 border-r border-paper-edge select-none">
                URL
              </span>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="youtube.com/watch?v=..."
                className="flex-1 bg-transparent font-mono text-[13px] text-ink-soft placeholder:text-ink-whisper outline-none"
                spellCheck={false}
              />
              {url && (
                <span
                  className={
                    "font-sans text-[11px] shrink-0 " +
                    (isValidUrl ? "text-accent" : "text-ink-ghost")
                  }
                >
                  {isValidUrl ? "✓ Valid" : "Invalid URL"}
                </span>
              )}
            </label>
          </div>

          {/* Thumbnail — slides in when URL is valid */}
          <div
            className={
              "overflow-hidden transition-all duration-300 " +
              (isValidUrl ? "max-h-[400px] opacity-100 mb-8" : "max-h-0 opacity-0 mb-0")
            }
          >
            <div className="aspect-video max-w-[420px] w-full mx-auto rounded-editorial overflow-hidden bg-gradient-to-br from-[#3B4870] to-[#8B6B4A] shadow-[0_1px_4px_rgba(42,37,32,0.1)]">
              {thumbnailUrl && (
                <img
                  src={thumbnailUrl}
                  alt="Video thumbnail"
                  className="w-full h-full object-cover"
                  onError={(e) => (e.currentTarget.style.display = "none")}
                />
              )}
            </div>
            <div className="flex items-center justify-center gap-2 mt-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-accent inline-block" />
              <span className="font-sans text-[11px] text-ink-ghost tracking-wide">
                Ready to process
              </span>
            </div>
          </div>

          {/* Explanation Depth Cards */}
          <div className="mb-10">
            <div className="eyebrow mb-4">Explanation depth</div>
            <div className="grid grid-cols-3 gap-3">
              {modes.map((m) => {
                const active = mode === m.key;
                return (
                  <button
                    key={m.key}
                    type="button"
                    onClick={() => setMode(m.key)}
                    className={
                      "relative flex flex-col items-start gap-3 p-4 rounded-editorial border text-left transition-all duration-150 " +
                      (active
                        ? "border-accent bg-accent/5 shadow-[0_2px_10px_rgba(74,91,140,0.12)]"
                        : "border-paper-edge bg-paper-soft hover:border-[#C4BAA5]")
                    }
                  >
                    {active && (
                      <span className="absolute top-3 right-3 font-sans text-[8px] text-accent tracking-[0.14em] uppercase">
                        {m.recommended ? "Recommended" : "Active"}
                      </span>
                    )}

                    <span
                      className={
                        "transition-colors " +
                        (active ? "text-accent" : "text-ink-ghost")
                      }
                    >
                      {m.icon}
                    </span>

                    <span>
                      <span className="block font-sans text-[11px] font-semibold text-ink tracking-[0.14em] uppercase">
                        {m.label}
                      </span>
                    </span>

                    <span className="font-sans text-[11px] text-ink-faded leading-relaxed">
                      {m.description}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* CTA */}
          <button
            onClick={handleSubmit}
            disabled={!isValidUrl}
            className={
              "btn-primary " +
              (isValidUrl ? "" : "opacity-40 cursor-not-allowed hover:bg-ink")
            }
          >
            Process video
            <span className="text-sm">→</span>
          </button>
        </div>
      </main>
    </div>
  );
}

/* ── Icons ───────────────────────────────────────────────────────────────── */

function LightbulbIcon() {
  return (
    <svg width="34" height="34" viewBox="0 0 36 36" fill="none"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 3C11.37 3 6 8.37 6 15C6 19.14 8.08 22.8 11.25 25.05V28.5H24.75V25.05C27.92 22.8 30 19.14 30 15C30 8.37 24.63 3 18 3Z" />
      <line x1="11.25" y1="30.5" x2="24.75" y2="30.5" />
      <line x1="12.75" y1="33"   x2="23.25" y2="33" />
      <path d="M14.5 28.5V22L18 17.5L21.5 22V28.5" />
      <line x1="14.5"  y1="25.5" x2="21.5"  y2="25.5" />
    </svg>
  );
}

function NeuralIcon() {
  const nodes = [
    [18,5],[8,10],[28,10],[5,19],[13,17],
    [18,19],[23,17],[31,19],[9,27],[18,29],[27,27],
  ];
  const edges = [
    [0,1],[0,2],[1,3],[1,4],[2,6],[2,7],
    [3,4],[3,8],[4,5],[5,6],[6,7],[7,10],
    [8,9],[9,10],[4,8],[6,10],[5,9],
  ];
  return (
    <svg width="34" height="34" viewBox="0 0 36 36" fill="none"
      stroke="currentColor" strokeWidth="1.3" strokeLinecap="round">
      {edges.map(([a, b], i) => (
        <line key={i}
          x1={nodes[a][0]} y1={nodes[a][1]}
          x2={nodes[b][0]} y2={nodes[b][1]}
        />
      ))}
      {nodes.map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="1.8" fill="currentColor" stroke="none" />
      ))}
    </svg>
  );
}

function CompassIcon() {
  return (
    <svg width="34" height="34" viewBox="0 0 36 36" fill="none"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="18" cy="5.5" r="2.5" />
      <line x1="18"   y1="8"  x2="18"   y2="12" />
      <line x1="18"   y1="12" x2="10.5" y2="32" />
      <line x1="18"   y1="12" x2="25.5" y2="32" />
      <line x1="13"   y1="21" x2="23"   y2="21" />
      <path d="M23.5 28.5 L25.5 32 L27.5 28.5" />
      <path d="M10.5 32 Q18 37.5 25.5 32" />
    </svg>
  );
}

/* ── Helper ──────────────────────────────────────────────────────────────── */

function extractYouTubeId(raw) {
  if (!raw || typeof raw !== "string") return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const patterns = [
    /(?:youtube\.com\/watch\?(?:.*&)?v=)([a-zA-Z0-9_-]{11})/,
    /(?:youtu\.be\/)([a-zA-Z0-9_-]{11})/,
    /(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})/,
    /(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
  ];
  for (const re of patterns) {
    const match = trimmed.match(re);
    if (match?.[1]) return match[1];
  }
  return null;
}
