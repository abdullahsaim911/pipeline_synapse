import { useState, useEffect } from "react";
import Sidebar from "./Sidebar";
import { getInterventionPoints } from "../api";

export default function Timeline({
  jobId,
  videoId,
  mode       = "standard",
  onOpenPoint = () => {},
  onNavigate  = () => {},
}) {
  const [data,           setData]           = useState(null);
  const [selectedId,     setSelectedId]     = useState(null);
  const [activeCategory, setActiveCategory] = useState("all");

  useEffect(() => {
    let cancelled = false;
    getInterventionPoints(jobId, videoId)
      .then((result) => {
        if (cancelled) return;
        setData(result);
        if (result.points.length > 0) setSelectedId(result.points[0].id);
      })
      .catch((err) => console.error("Failed to load points:", err));
    return () => { cancelled = true; };
  }, [jobId]);

  if (!data) {
    return (
      <div className="flex h-full w-full bg-paper">
        <Sidebar active="home" onNavigate={onNavigate} />
        <main className="flex-1 flex items-center justify-center">
          <div className="font-serif text-ink-quiet">Loading…</div>
        </main>
      </div>
    );
  }

  const totalSeconds     = parseTimeToSeconds(data.duration);
  const pointsByCategory = {
    equation: data.points.filter((p) => p.category === "equation"),
    diagram:  data.points.filter((p) => p.category === "diagram"),
    chart:    data.points.filter((p) => p.category === "chart"),
    graph:    data.points.filter((p) => p.category === "graph"),
  };
  const counts = {
    equation: pointsByCategory.equation.length,
    diagram:  pointsByCategory.diagram.length,
    chart:    pointsByCategory.chart.length,
    graph:    pointsByCategory.graph.length,
    all:      data.points.length,
  };
  const visiblePoints =
    activeCategory === "all"
      ? data.points
      : data.points.filter((p) => p.category === activeCategory);

  const selectedPoint = data.points.find((p) => p.id === selectedId);
  const thumbnailUrl  = videoId
    ? `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`
    : null;
  const frameUrl = selectedPoint?._frame_path
    ? `http://127.0.0.1:8000/${selectedPoint._frame_path.replace(/^\.\//, "")}`
    : null;

  return (
    <div className="flex h-full w-full bg-paper overflow-hidden">
      <Sidebar active="home" onNavigate={onNavigate} />

      {/* ── Left pane ─────────────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 overflow-y-auto border-r border-paper-edge">
        <div className="px-10 py-10">

          {/* Breadcrumb */}
          <div className="font-sans text-[10px] text-ink-ghost uppercase tracking-[0.14em] mb-7">
            Home
            <span className="mx-2 text-paper-rule">›</span>
            <span className="text-ink-muted">{data.videoTitle}</span>
          </div>

          {/* ── Compact video header ── */}
          <div className="flex gap-6 items-start mb-9 pb-8 border-b border-paper-edge">
            {/* Small thumbnail */}
            <div className="w-[180px] shrink-0 aspect-video rounded-editorial overflow-hidden relative bg-gradient-to-br from-[#3B4870] to-[#8B6B4A] shadow-[0_1px_4px_rgba(42,37,32,0.1)]">
              {thumbnailUrl && (
                <img
                  src={thumbnailUrl}
                  alt="thumbnail"
                  className="w-full h-full object-cover"
                  onError={(e) => (e.currentTarget.style.display = "none")}
                />
              )}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-8 h-8 rounded-full bg-paper/85 flex items-center justify-center">
                  <span className="text-ink text-[11px] ml-0.5">▶</span>
                </div>
              </div>
              <div className="absolute bottom-1.5 right-1.5 font-mono text-[9px] bg-ink/75 text-paper px-1.5 py-0.5 rounded-sm">
                {data.duration}
              </div>
            </div>

            {/* Title + meta + stats */}
            <div className="flex-1 min-w-0">
              <h1 className="headline text-[22px] leading-snug mb-1 truncate">
                {data.videoTitle}
              </h1>
              <div className="font-sans text-[11px] text-ink-ghost mb-5">
                {data.author}
              </div>

              {/* Stat pills */}
              <div className="flex flex-wrap gap-2">
                {[
                  { label: "All",       count: counts.all,      color: "#4A5B8C" },
                  { label: "Equations", count: counts.equation,  color: CATEGORY_COLORS.equation },
                  { label: "Diagrams",  count: counts.diagram,   color: CATEGORY_COLORS.diagram  },
                  { label: "Charts",    count: counts.chart,     color: CATEGORY_COLORS.chart    },
                  { label: "Graphs",    count: counts.graph,     color: CATEGORY_COLORS.graph    },
                ].map((s) => (
                  <span
                    key={s.label}
                    className="inline-flex items-center gap-1.5 font-sans text-[11px] px-2.5 py-1 rounded-full border border-paper-edge bg-paper-soft"
                  >
                    <span
                      className="w-1.5 h-1.5 rounded-full"
                      style={{ backgroundColor: s.color }}
                    />
                    <span className="text-ink-muted">{s.count}</span>
                    <span className="text-ink-ghost">{s.label}</span>
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* ── Timeline ── */}
          <div className="mb-6">
            <div className="eyebrow mb-4">Intervention timeline</div>
            <div className="bg-paper-soft border border-paper-edge rounded-editorial p-6 pb-4">
              <LaneTimeline
                totalSeconds={totalSeconds}
                pointsByCategory={pointsByCategory}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            </div>
          </div>

          {/* ── Category filter ── */}
          <div className="flex gap-2 mb-5">
            {[
              { key: "all",      label: "All",       count: counts.all      },
              { key: "equation", label: "Equations", count: counts.equation  },
              { key: "diagram",  label: "Diagrams",  count: counts.diagram   },
              { key: "chart",    label: "Charts",    count: counts.chart     },
              { key: "graph",    label: "Graphs",    count: counts.graph     },
            ].map((cat) => {
              const active = activeCategory === cat.key;
              return (
                <button
                  key={cat.key}
                  onClick={() => setActiveCategory(cat.key)}
                  className={
                    "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full font-sans text-[11px] border transition-all " +
                    (active
                      ? "bg-ink text-paper border-ink"
                      : "bg-paper-soft text-ink-faded border-paper-edge hover:border-[#C4BAA5] hover:text-ink")
                  }
                >
                  {cat.label}
                  <span className={active ? "text-paper/60" : "text-ink-ghost"}>
                    {cat.count}
                  </span>
                </button>
              );
            })}
          </div>

          {/* ── Point list — vertical ── */}
          <div className="space-y-1.5">
            {visiblePoints.map((point) => (
              <PointRow
                key={point.id}
                point={point}
                isSelected={point.id === selectedId}
                onClick={() => setSelectedId(point.id)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* ── Right pane — preview ──────────────────────────────────────────── */}
      <div className="w-[300px] shrink-0 overflow-y-auto bg-paper-soft">
        {selectedPoint ? (
          <PreviewPanel
            point={selectedPoint}
            thumbnailUrl={thumbnailUrl}
            frameUrl={frameUrl}
            onOpen={() => onOpenPoint(selectedPoint)}
          />
        ) : (
          <div className="h-full flex flex-col items-center justify-center px-8 text-center">
            <div className="w-10 h-10 rounded-full bg-paper-edge flex items-center justify-center mb-4">
              <span className="text-ink-ghost text-sm">↑</span>
            </div>
            <p className="font-sans text-[12px] text-ink-ghost leading-relaxed">
              Select a marker on the timeline or a point below to preview it here.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Preview panel ───────────────────────────────────────────────────────── */

function PreviewPanel({ point, thumbnailUrl, frameUrl, onOpen }) {
  const color = CATEGORY_COLORS[point.category];
  const description = CATEGORY_DESCRIPTIONS[point.category];

  return (
    <div className="p-7">
      {/* Category + timestamp row */}
      <div className="flex items-center justify-between mb-5">
        <span
          className="inline-flex items-center gap-1.5 font-sans text-[10px] uppercase tracking-[0.16em]"
          style={{ color }}
        >
          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
          {point.category}
        </span>
        <span className="font-mono text-[11px] text-ink-ghost">
          {point.timestamp}
        </span>
      </div>

      {/* Title */}
      <h2 className="font-serif text-[22px] text-ink leading-snug mb-5">
        {point.title}
      </h2>

      {/* Frame preview */}
      <div className="aspect-video rounded-editorial overflow-hidden bg-gradient-to-br from-[#3B4870] to-[#8B6B4A] mb-5 relative">
        {frameUrl ? (
          <img
            src={frameUrl}
            alt="frame"
            className="w-full h-full object-cover"
            onError={(e) => {
              e.currentTarget.style.display = "none";
              const fallback = e.currentTarget.parentElement.querySelector(".fallback-img");
              if (fallback) fallback.style.display = "block";
            }}
          />
        ) : null}
        {thumbnailUrl && (
          <img
            src={thumbnailUrl}
            alt="thumbnail"
            className="fallback-img w-full h-full object-cover"
            style={{ display: frameUrl ? "none" : "block" }}
            onError={(e) => (e.currentTarget.style.display = "none")}
          />
        )}
        {/* Timestamp overlay */}
        <div className="absolute bottom-2 right-2 font-mono text-[9px] bg-ink/75 text-paper px-1.5 py-0.5 rounded-sm">
          {point.timestamp}
        </div>
      </div>

      {/* Description */}
      <p className="font-sans text-[12px] text-ink-faded leading-relaxed mb-7">
        {description}
      </p>

      {/* Divider */}
      <div className="border-t border-paper-edge mb-7" />

      {/* CTA */}
      <button
        onClick={onOpen}
        className="btn-primary w-full justify-center"
      >
        Open explanation
        <span className="text-sm">→</span>
      </button>
    </div>
  );
}

/* ── Point row ───────────────────────────────────────────────────────────── */

function PointRow({ point, isSelected, onClick }) {
  const color = CATEGORY_COLORS[point.category];
  return (
    <button
      onClick={onClick}
      className={
        "w-full flex items-center gap-4 px-4 py-3 rounded-editorial text-left transition-all border " +
        (isSelected
          ? "bg-paper border-paper-rule shadow-[0_1px_4px_rgba(42,37,32,0.07)]"
          : "bg-paper-soft border-transparent hover:border-paper-edge")
      }
    >
      {/* Color accent bar */}
      <span
        className="w-0.5 h-8 rounded-full shrink-0"
        style={{ backgroundColor: isSelected ? color : "#DDD5C2" }}
      />

      {/* Content */}
      <span className="flex-1 min-w-0">
        <span
          className="block font-sans text-[10px] uppercase tracking-[0.14em] mb-0.5"
          style={{ color }}
        >
          {point.category}
        </span>
        <span className="block font-serif text-[14px] text-ink leading-snug truncate">
          {point.title}
        </span>
      </span>

      {/* Timestamp */}
      <span className="font-mono text-[11px] text-ink-ghost shrink-0">
        {point.timestamp}
      </span>

      {/* Arrow on selected */}
      {isSelected && (
        <span className="text-ink-ghost text-[11px] shrink-0">›</span>
      )}
    </button>
  );
}

/* ── Lane Timeline (SVG) ─────────────────────────────────────────────────── */

function LaneTimeline({ totalSeconds, pointsByCategory, selectedId, onSelect }) {
  const PADDING_LEFT   = 72;
  const PADDING_RIGHT  = 20;
  const VIEWBOX_WIDTH  = 760;
  const TIMELINE_WIDTH = VIEWBOX_WIDTH - PADDING_LEFT - PADDING_RIGHT;
  const LANE_H         = 48; // height of each lane band

  const LANES = {
    equation: 55,
    diagram:  55 + LANE_H + 12,
    chart:    55 + (LANE_H + 12) * 2,
    graph:    55 + (LANE_H + 12) * 3,
  };
  const SVG_HEIGHT = 55 + (LANE_H + 12) * 4 - 4;

  const timeToX = (s) => PADDING_LEFT + (s / totalSeconds) * TIMELINE_WIDTH;

  // Ruler labels — every minute up to 6 labels
  const numLabels  = Math.min(6, Math.ceil(totalSeconds / 60) + 1);
  const labelStep  = totalSeconds / (numLabels - 1);
  const labels     = Array.from({ length: numLabels }, (_, i) => {
    const secs = Math.round(i * labelStep);
    return { secs, label: secondsToTime(secs), x: timeToX(secs) };
  });

  // All points for playhead lookup
  const allPoints = [
    ...pointsByCategory.equation,
    ...pointsByCategory.diagram,
    ...pointsByCategory.chart,
  ];
  const selectedPoint = allPoints.find((p) => p.id === selectedId);
  const playheadX     = selectedPoint
    ? timeToX(parseTimeToSeconds(selectedPoint.timestamp))
    : null;

  const LANES_LIST = [
    { key: "equation", label: "Equations", color: CATEGORY_COLORS.equation },
    { key: "diagram",  label: "Diagrams",  color: CATEGORY_COLORS.diagram  },
    { key: "chart",    label: "Charts",    color: CATEGORY_COLORS.chart    },
    { key: "graph",    label: "Graphs",    color: CATEGORY_COLORS.graph    },
  ];

  return (
    <svg viewBox={`0 0 ${VIEWBOX_WIDTH} ${SVG_HEIGHT}`} width="100%" className="block overflow-visible">

      {/* ── Ruler ── */}
      <line
        x1={PADDING_LEFT} y1={18}
        x2={VIEWBOX_WIDTH - PADDING_RIGHT} y2={18}
        stroke="#C4BAA5" strokeWidth="0.5"
      />
      {labels.map((l, i) => (
        <g key={i}>
          <line
            x1={l.x} y1={14} x2={l.x} y2={22}
            stroke="#C4BAA5" strokeWidth="0.5"
          />
          <text
            x={l.x} y={10}
            textAnchor={i === 0 ? "start" : i === labels.length - 1 ? "end" : "middle"}
            fontFamily="-apple-system, sans-serif" fontSize="8"
            fill="#8A8070" letterSpacing="0.06em"
          >
            {l.label}
          </text>
        </g>
      ))}

      {/* ── Playhead ── */}
      {playheadX && (
        <line
          x1={playheadX} y1={22}
          x2={playheadX} y2={SVG_HEIGHT}
          stroke="#4A5B8C" strokeWidth="1"
          strokeDasharray="3 3" opacity="0.5"
        />
      )}

      {/* ── Lanes ── */}
      {LANES_LIST.map(({ key, label, color }) => {
        const y = LANES[key];
        return (
          <g key={key}>
            {/* Lane band */}
            <rect
              x={PADDING_LEFT} y={y - LANE_H / 2}
              width={TIMELINE_WIDTH} height={LANE_H}
              fill={color} fillOpacity="0.05"
              rx="3"
            />
            {/* Lane center line */}
            <line
              x1={PADDING_LEFT} y1={y}
              x2={VIEWBOX_WIDTH - PADDING_RIGHT} y2={y}
              stroke={color} strokeWidth="0.5" strokeOpacity="0.3"
            />
            {/* Lane label */}
            <text
              x={0} y={y + 4}
              fontFamily="-apple-system, sans-serif" fontSize="9"
              fill={color} letterSpacing="0.1em"
              style={{ textTransform: "uppercase" }}
            >
              {label}
            </text>
            {/* Markers */}
            {pointsByCategory[key].map((point) => (
              <Marker
                key={point.id}
                x={timeToX(parseTimeToSeconds(point.timestamp))}
                y={y}
                color={color}
                isSelected={point.id === selectedId}
                timestamp={point.timestamp}
                onClick={() => onSelect(point.id)}
              />
            ))}
          </g>
        );
      })}
    </svg>
  );
}

/* ── Marker ──────────────────────────────────────────────────────────────── */

function Marker({ x, y, color, isSelected, timestamp, onClick }) {
  const size = isSelected ? 12 : 7;

  return (
    <g onClick={onClick} style={{ cursor: "pointer" }}>
      {/* Hit area */}
      <circle cx={x} cy={y} r={16} fill="transparent" />

      {/* Pulse ring on selected */}
      {isSelected && (
        <circle cx={x} cy={y} r={size + 4} fill="none" stroke={color} strokeWidth="1" opacity="0">
          <animate attributeName="r"       values={`${size};${size + 12}`} dur="2s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.6;0"                  dur="2s" repeatCount="indefinite" />
        </circle>
      )}

      {/* Dot */}
      <circle
        cx={x} cy={y} r={size}
        fill={isSelected ? color : "#FDFBF5"}
        stroke={color}
        strokeWidth={isSelected ? 0 : 1.5}
      />

      {/* Checkmark on selected */}
      {isSelected && (
        <text x={x} y={y + 4.5} textAnchor="middle"
          fontFamily="-apple-system, sans-serif" fontSize="10" fill="#fff" fontWeight="600">
          ✓
        </text>
      )}

      {/* Timestamp — only on selected */}
      {isSelected && (
        <text x={x} y={y + 26} textAnchor="middle"
          fontFamily="-apple-system, sans-serif" fontSize="8"
          fill={color} fontWeight="500" letterSpacing="0.04em">
          {timestamp}
        </text>
      )}
    </g>
  );
}


/* ── Constants ───────────────────────────────────────────────────────────── */

const CATEGORY_COLORS = {
  equation: "#6B5F8C",
  diagram:  "#8B6B4A",
  chart:    "#8C5A47",
  graph:    "#4A8C6B",
};

const CATEGORY_DESCRIPTIONS = {
  equation: "A mathematical expression or formula appears at this moment. Open the explanation to get a step-by-step breakdown of what it means and why it matters in context.",
  diagram:  "A visual diagram is shown here. Open the explanation to understand the relationships and concepts it illustrates.",
  chart:    "A data chart or graph appears at this point. Open the explanation to learn what the data represents and its significance.",
  graph:    "A graph or plot appears here. Open the explanation to understand what the axes represent and what the shape of the graph tells you.",
};

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function parseTimeToSeconds(timeStr) {
  const parts = timeStr.split(":").map(Number);
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  return 0;
}

function secondsToTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
