import { useState, useEffect, useRef } from "react";
import Sidebar from "./Sidebar";

export default function Processing({
  jobId,
  videoId,
  mode = "standard",
  promise,
  onDone = () => {},
  onCancel = () => {},
  onNavigate = () => {},
}) {
  const [status, setStatus] = useState({
    stage: "downloading",
    progress: 0,
    message: "Starting…",
    done: false,
  });
  const [nodeFlicker, setNodeFlicker] = useState(false);

  const startTimeRef    = useRef(Date.now());
  const prevProgressRef = useRef(0);

  // ── Progress polling ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!promise) return;

    const ESTIMATED_MS = 90_000;
    const interval = setInterval(() => {
      const elapsed  = Date.now() - startTimeRef.current;
      const progress = Math.min(95, Math.round((elapsed / ESTIMATED_MS) * 100));
      let stage = "downloading";
      if (progress >= 20) stage = "extracting";
      if (progress >= 45) stage = "detecting";
      if (progress >= 80) stage = "generating";
      setStatus((s) => ({ ...s, progress, stage }));
    }, 1000);

    promise
      .then((result) => {
        clearInterval(interval);
        setStatus({ stage: "generating", progress: 100, message: "Complete", done: true });
        setTimeout(() => onDone(result.jobId ?? jobId), 3000);
      })
      .catch(() => {
        clearInterval(interval);
        alert("Something went wrong. Please try again.");
        onCancel();
      });

    return () => clearInterval(interval);
  }, [promise]);

  // ── Node flicker on each progress tick ────────────────────────────────────
  useEffect(() => {
    if (status.progress === prevProgressRef.current) return;
    prevProgressRef.current = status.progress;
    setNodeFlicker(true);
    const t = setTimeout(() => setNodeFlicker(false), 350);
    return () => clearTimeout(t);
  }, [status.progress]);

  // ── Derived values ────────────────────────────────────────────────────────
  const remainingSecs    = computeRemainingSecs(startTimeRef.current, status.progress);
  const thumbnailUrl     = videoId
    ? `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`
    : null;

  const stages = [
    { key: "downloading", title: "Fetching the video" },
    { key: "extracting",  title: "Scanning each frame" },
    { key: "detecting",   title: "Spotting equations, diagrams & charts" },
    { key: "generating",  title: "Generating your explanations" },
  ];
  const currentStageIndex = stages.findIndex((s) => s.key === status.stage);

  // ── SVG ring geometry ─────────────────────────────────────────────────────
  const RING_SIZE        = 210;
  const TRACK_STROKE     = 2;
  const PROGRESS_STROKE  = 7;
  const RING_RADIUS      = (RING_SIZE - PROGRESS_STROKE) / 2;
  const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;
  const ringOffset       = RING_CIRCUMFERENCE * (1 - status.progress / 100);

  // ── Pulse ring config: [diameter, delay] ─────────────────────────────────
  const pulseRings = [
    { size: 18, delay: "0s"    },
    { size: 32, delay: "0.6s"  },
    { size: 48, delay: "1.2s"  },
    { size: 64, delay: "1.8s"  },
  ];

  return (
    <div className="flex h-full w-full bg-paper overflow-hidden">
      <Sidebar active="home" onNavigate={onNavigate} />

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-[720px] px-20 py-14">

          {/* Eyebrow */}
          <div className="eyebrow mb-3.5">
            {status.done ? "Complete" : "Neural analysis"}
          </div>

          {/* Headline */}
          <h1 className="headline text-[36px]">
            {status.done ? "Ready when you are." : "Studying your video."}
          </h1>
          <p className="subtitle text-[17px] mt-3 mb-10">
            {status.done
              ? "Every visual moment has been identified and explained."
              : "Extracting visual learning moments frame by frame."}
          </p>

          {/* ── Main row: neural loader + video info ── */}
          <div className="grid grid-cols-[auto_1fr] gap-12 items-center mb-14 py-8 border-t border-b border-paper-edge">

            {/* Neural Loader */}
            <div
              className="relative flex-shrink-0"
              style={{ width: RING_SIZE, height: RING_SIZE }}
            >
              {/* Expanding pulse rings behind the SVG */}
              {!status.done && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  {pulseRings.map((ring, i) => (
                    <div
                      key={i}
                      className="absolute rounded-full bg-accent/[0.12] animate-ping"
                      style={{
                        width:           ring.size,
                        height:          ring.size,
                        animationDuration: "2.4s",
                        animationDelay:  ring.delay,
                        animationTimingFunction: "cubic-bezier(0.4,0,0.2,1)",
                      }}
                    />
                  ))}
                </div>
              )}

              {/* Progress ring SVG */}
              <svg
                width={RING_SIZE}
                height={RING_SIZE}
                className="transform -rotate-90"
              >
                {/* Track */}
                <circle
                  cx={RING_SIZE / 2}
                  cy={RING_SIZE / 2}
                  r={RING_RADIUS}
                  fill="none"
                  stroke="#E5DFD0"
                  strokeWidth={TRACK_STROKE}
                />
                {/* Progress arc */}
                <circle
                  cx={RING_SIZE / 2}
                  cy={RING_SIZE / 2}
                  r={RING_RADIUS}
                  fill="none"
                  stroke="#4A5B8C"
                  strokeWidth={PROGRESS_STROKE}
                  strokeLinecap="round"
                  strokeDasharray={RING_CIRCUMFERENCE}
                  strokeDashoffset={ringOffset}
                  style={{
                    transition: "stroke-dashoffset 900ms cubic-bezier(0.4,0,0.2,1)",
                  }}
                />
              </svg>

              {/* Center content */}
              <div className="absolute inset-0 flex flex-col items-center justify-center">

                {/* Central node with flicker */}
                <div
                  className={
                    "w-2.5 h-2.5 rounded-full mb-3 transition-all duration-300 " +
                    (nodeFlicker
                      ? "bg-accent scale-150 shadow-[0_0_10px_3px_rgba(74,91,140,0.6)]"
                      : "bg-accent/60")
                  }
                />

                {/* Percentage */}
                <div className="font-serif text-[46px] text-ink leading-none tracking-editorial">
                  {status.progress}
                  <span className="text-ink-ghost text-[22px] align-top ml-0.5">%</span>
                </div>

                {/* Time remaining */}
                <div className="font-mono text-[9px] text-ink-ghost tracking-[0.16em] uppercase mt-2">
                  {status.done
                    ? "Complete"
                    : remainingSecs !== null
                    ? `${formatMmSs(remainingSecs)} remaining`
                    : "Estimating…"}
                </div>
              </div>
            </div>

            {/* Video info */}
            <div>
              <div className="aspect-video w-full max-w-[280px] rounded-editorial overflow-hidden relative bg-gradient-to-br from-[#3B4870] to-[#8B6B4A] mb-4">
                {thumbnailUrl && (
                  <img
                    src={thumbnailUrl}
                    alt="Video thumbnail"
                    className="w-full h-full object-cover"
                    onError={(e) => (e.currentTarget.style.display = "none")}
                  />
                )}
              </div>
              <div className="font-serif text-[18px] text-ink leading-tight">
                Your video
              </div>
              <div className="font-sans text-[11px] text-ink-ghost mt-1.5 tracking-wide capitalize">
                {mode} depth
              </div>
            </div>
          </div>

          {/* ── Stage list ── */}
          <div className="space-y-1">
            {stages.map((stage, index) => {
              const isComplete = index < currentStageIndex || status.done;
              const isActive   = index === currentStageIndex && !status.done;
              const isPending  = index > currentStageIndex && !status.done;

              return (
                <div
                  key={stage.key}
                  className={
                    "grid grid-cols-[28px_1fr_auto] gap-4 items-center py-3.5 transition-opacity duration-500 " +
                    (isPending ? "opacity-30" : "opacity-100")
                  }
                >
                  {/* Status indicator */}
                  <div className="flex items-center justify-center">
                    {isComplete && (
                      <div className="w-5 h-5 rounded-full bg-accent flex items-center justify-center">
                        <span className="text-paper text-[10px] font-bold">✓</span>
                      </div>
                    )}
                    {isActive && (
                      <div className="relative w-5 h-5 flex items-center justify-center">
                        <div
                          className="absolute w-5 h-5 rounded-full bg-accent/30 animate-ping"
                          style={{ animationDuration: "1.5s" }}
                        />
                        <div className="w-2.5 h-2.5 rounded-full bg-accent" />
                      </div>
                    )}
                    {isPending && (
                      <div className="w-2 h-2 rounded-full bg-paper-rule" />
                    )}
                  </div>

                  {/* Stage title */}
                  <div
                    className={
                      "font-serif text-[16px] " +
                      (isComplete ? "text-ink-ghost" : "text-ink")
                    }
                  >
                    {stage.title}
                  </div>

                  {/* Right label */}
                  <div className="font-sans text-[10px] uppercase tracking-[0.14em]">
                    {isComplete && <span className="text-ink-ghost">Done</span>}
                    {isActive   && <span className="text-accent font-medium">Now</span>}
                    {isPending  && <span className="text-ink-whisper">Next</span>}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Cancel */}
          {!status.done && (
            <button
              onClick={onCancel}
              className="mt-10 px-5 py-2 bg-transparent border border-paper-edge rounded-editorial font-sans text-[12px] text-ink-faded hover:text-ink hover:border-[#C4BAA5] transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      </main>
    </div>
  );
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function computeRemainingSecs(startTime, progress) {
  if (progress < 5 || progress >= 100) return null;
  const elapsedMs       = Date.now() - startTime;
  const totalEstimateMs = (elapsedMs / progress) * 100;
  const remainingMs     = totalEstimateMs - elapsedMs;
  return Math.max(1, Math.round(remainingMs / 1000));
}

function formatMmSs(secs) {
  const m = Math.floor(secs / 60).toString().padStart(2, "0");
  const s = (secs % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}
