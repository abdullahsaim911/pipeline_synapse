import { useState, useEffect } from "react";
import Sidebar from "./Sidebar";
import { getLibrary, removeFromLibrary } from "../storage";

const CATEGORY_COLORS = {
  equation: "#6B5F8C",
  diagram:  "#8B6B4A",
  chart:    "#8C5A47",
  graph:    "#4A8C6B",
};

export default function LibraryScreen({
  onNavigate = () => {},
  onOpenExplanation = () => {},
}) {
  const [library, setLibrary] = useState([]);
  const [search, setSearch] = useState("");

  useEffect(() => {
    setLibrary(getLibrary());
  }, []);

  const handleRemove = (id) => {
    removeFromLibrary(id);
    setLibrary(getLibrary());
  };

  const query = search.toLowerCase().trim();
  const visible = query
    ? library.filter(
        (item) =>
          (item.title || "").toLowerCase().includes(query) ||
          (item.videoTitle || "").toLowerCase().includes(query) ||
          (item.category || "").toLowerCase().includes(query),
      )
    : library;

  return (
    <div className="flex h-full w-full bg-paper overflow-hidden">
      <Sidebar active="library" onNavigate={onNavigate} />

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-[680px] px-10 py-12">
          {/* ── Header ── */}
          <h1 className="font-serif text-[36px] text-ink leading-none mb-6">
            My Library
          </h1>

          {/* ── Search ── */}
          <div className="relative mb-8">
            <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-ghost pointer-events-none">
              <SearchIcon />
            </span>
            <input
              type="text"
              placeholder="Filter by topic or video title…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full font-sans text-[14px] text-ink bg-paper-soft border border-paper-edge rounded-editorial pl-10 pr-9 py-3 placeholder-ink-ghost focus:outline-none focus:border-[#C4BAA5] transition-colors"
            />
            {search && (
              <button
                onClick={() => setSearch("")}
                className="absolute right-3.5 top-1/2 -translate-y-1/2 text-ink-ghost hover:text-ink text-lg leading-none"
              >
                ×
              </button>
            )}
          </div>

          {/* ── Section label ── */}
          {library.length > 0 && (
            <div className="flex items-center gap-3 mb-5">
              <span className="font-sans text-[13px] font-medium text-ink">
                Recent Explanations
              </span>
              <span className="font-mono text-[11px] bg-ink text-paper px-2.5 py-0.5 rounded-full">
                {library.length}
              </span>
            </div>
          )}

          {/* ── List ── */}
          {library.length === 0 ? (
            <EmptyState />
          ) : visible.length === 0 ? (
            <p className="font-sans text-[13px] text-ink-ghost mt-2">
              No results for &ldquo;{search}&rdquo;.
            </p>
          ) : (
            <div className="space-y-3">
              {visible.map((item) => (
                <ExplanationCard
                  key={item.id}
                  item={item}
                  onReadText={() =>
                    onOpenExplanation({ ...item, autoPlay: false })
                  }
                  onPlayAudio={() =>
                    onOpenExplanation({ ...item, autoPlay: true })
                  }
                  onRemove={() => handleRemove(item.id)}
                />
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

/* ── Compact horizontal card ─────────────────────────────────────────────── */

function ExplanationCard({ item, onReadText, onPlayAudio, onRemove }) {
  const color = CATEGORY_COLORS[item.category] || "#4A5B8C";
  const thumbnailUrl = item.videoId
    ? `https://i.ytimg.com/vi/${item.videoId}/hqdefault.jpg`
    : null;
  const dateStr = item.savedAt
    ? new Date(item.savedAt).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      })
    : "";

  return (
    <div className="flex gap-4 p-4 bg-paper-soft border border-paper-edge rounded-editorial hover:border-[#C4BAA5] transition-colors group">
      {/* Thumbnail — small, fixed width */}
      <div className="w-[120px] shrink-0 aspect-video rounded-lg overflow-hidden bg-gradient-to-br from-[#3B4870] to-[#8B6B4A] relative">
        {thumbnailUrl && (
          <img
            src={thumbnailUrl}
            alt="frame"
            className="w-full h-full object-cover"
            onError={(e) => (e.currentTarget.style.display = "none")}
          />
        )}
        {/* Timestamp chip */}
        <div className="absolute bottom-1 right-1 font-mono text-[8px] bg-ink/75 text-paper px-1 py-0.5 rounded-sm">
          {item.timestamp}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Category + date */}
        <div className="flex items-center gap-1.5 mb-1">
          <span
            className="font-sans text-[9px] uppercase tracking-[0.14em] font-medium"
            style={{ color }}
          >
            {item.category}
          </span>
          <span className="text-paper-rule text-[9px]">·</span>
          <span className="font-sans text-[10px] text-ink-ghost">
            {dateStr}
          </span>
          {/* Remove button — shows on hover */}
          <button
            onClick={onRemove}
            className="ml-auto text-ink-ghost hover:text-ink text-[11px] opacity-0 group-hover:opacity-100 transition-opacity"
          >
            ×
          </button>
        </div>

        {/* Title */}
        <h3 className="font-serif text-[15px] text-ink leading-snug line-clamp-2 mb-1">
          {item.title}
        </h3>

        {/* Video title */}
        <p className="font-sans text-[11px] text-ink-ghost truncate mb-3">
          {item.videoTitle}
        </p>

        {/* Buttons */}
        <div className="flex gap-2 mt-auto">
          <button
            onClick={onReadText}
            className="flex items-center gap-1.5 bg-ink text-paper font-sans text-[11px] font-medium uppercase tracking-[0.07em] px-3 py-1.5 rounded-lg hover:bg-[#0f0c08] transition-colors"
          >
            <DocIcon />
            Read text
          </button>
          <button
            onClick={onPlayAudio}
            className="flex items-center gap-1.5 border border-paper-edge text-ink font-sans text-[11px] font-medium uppercase tracking-[0.07em] px-3 py-1.5 rounded-lg hover:border-[#C4BAA5] hover:bg-paper transition-colors"
          >
            <PlayIcon />
            Play audio
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Empty state ─────────────────────────────────────────────────────────── */

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="w-12 h-12 rounded-full bg-paper-edge flex items-center justify-center mb-4">
        <span className="text-ink-ghost text-lg">▶</span>
      </div>
      <p className="font-sans text-[13px] text-ink-ghost max-w-[260px] leading-relaxed">
        Open any explanation from the Timeline and it will be saved here
        automatically.
      </p>
    </div>
  );
}

/* ── Icons ───────────────────────────────────────────────────────────────── */

function SearchIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    >
      <circle cx="6.5" cy="6.5" r="4.5" />
      <line x1="10" y1="10" x2="14" y2="14" />
    </svg>
  );
}

function DocIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M4 2h6l3 3v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z" />
      <polyline points="10,2 10,5 13,5" />
      <line x1="5" y1="9" x2="11" y2="9" />
      <line x1="5" y1="12" x2="9" y2="12" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="8" cy="8" r="7" />
      <polygon
        points="6.5,5.5 11.5,8 6.5,10.5"
        fill="currentColor"
        stroke="none"
      />
    </svg>
  );
}
