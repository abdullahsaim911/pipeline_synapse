import { useState, useEffect } from "react";
import Sidebar from "./Sidebar";
import { getBookmarks, removeBookmark } from "../storage";

const CATEGORY_COLORS = {
  equation: "#6B5F8C",
  diagram:  "#8B6B4A",
  chart:    "#8C5A47",
  graph:    "#4A8C6B",
};

export default function BookmarksScreen({
  onNavigate     = () => {},
  onOpenBookmark = () => {},
}) {
  const [bookmarks, setBookmarks] = useState([]);

  useEffect(() => { setBookmarks(getBookmarks()); }, []);

  const handleRemove = (id) => {
    removeBookmark(id);
    setBookmarks(getBookmarks());
  };

  return (
    <div className="flex h-full w-full bg-paper overflow-hidden">
      <Sidebar active="bookmarks" onNavigate={onNavigate} />

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-[700px] px-12 py-12">

          <div className="eyebrow mb-3">Bookmarks</div>
          <h1 className="headline text-[32px] mb-2">Your saved explanations.</h1>
          <p className="subtitle text-[15px] mb-10">
            {bookmarks.length === 0
              ? "Nothing saved yet. Bookmark any explanation while reading it."
              : `${bookmarks.length} explanation${bookmarks.length === 1 ? "" : "s"} saved.`}
          </p>

          {bookmarks.length === 0 ? (
            <EmptyState
              icon="❏"
              message="Open any explanation and click Bookmark to save it here."
            />
          ) : (
            <div className="space-y-3">
              {bookmarks.map((bm) => (
                <BookmarkCard
                  key={bm.id}
                  bookmark={bm}
                  onOpen={() => onOpenBookmark(bm)}
                  onRemove={() => handleRemove(bm.id)}
                />
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

/* ── Bookmark card ───────────────────────────────────────────────────────── */

function BookmarkCard({ bookmark, onOpen, onRemove }) {
  const color = CATEGORY_COLORS[bookmark.category] || "#4A5B8C";
  const date  = new Date(bookmark.savedAt).toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric",
  });

  return (
    <div className="flex items-start gap-4 p-5 bg-paper-soft border border-paper-edge rounded-editorial hover:border-[#C4BAA5] transition-colors">

      {/* Color bar */}
      <span
        className="w-0.5 h-14 rounded-full shrink-0 mt-1"
        style={{ backgroundColor: color }}
      />

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5">
          <span
            className="font-sans text-[10px] uppercase tracking-[0.14em]"
            style={{ color }}
          >
            {bookmark.category}
          </span>
          <span className="text-paper-rule">·</span>
          <span className="font-mono text-[10px] text-ink-ghost">
            {bookmark.timestamp}
          </span>
          <span className="text-paper-rule">·</span>
          <span className="font-sans text-[10px] text-ink-ghost truncate">
            {bookmark.videoTitle}
          </span>
        </div>

        <div className="font-serif text-[17px] text-ink leading-snug mb-1">
          {bookmark.title}
        </div>

        {bookmark.explanation?.subtitle && (
          <div className="font-sans text-[12px] text-ink-faded leading-relaxed line-clamp-2">
            {bookmark.explanation.subtitle}
          </div>
        )}

        <div className="font-sans text-[10px] text-ink-ghost mt-2 uppercase tracking-[0.08em]">
          Saved {date} · {bookmark.mode} depth
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-col items-end gap-2 shrink-0">
        <button onClick={onOpen} className="btn-primary text-[12px] px-4 py-2">
          Open <span className="text-sm">→</span>
        </button>
        <button
          onClick={onRemove}
          className="font-sans text-[11px] text-ink-ghost hover:text-ink transition-colors"
        >
          Remove
        </button>
      </div>
    </div>
  );
}

/* ── Empty state ─────────────────────────────────────────────────────────── */

function EmptyState({ icon, message }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="w-12 h-12 rounded-full bg-paper-edge flex items-center justify-center mb-4">
        <span className="text-ink-ghost text-lg">{icon}</span>
      </div>
      <p className="font-sans text-[13px] text-ink-ghost max-w-[260px] leading-relaxed">
        {message}
      </p>
    </div>
  );
}
