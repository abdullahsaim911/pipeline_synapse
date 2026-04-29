import { useState, useEffect } from "react";
import { getBookmarks, getLibrary } from "../storage";

export default function Sidebar({ active = "home", onNavigate = () => {} }) {
  const [bookmarkCount, setBookmarkCount] = useState(0);
  const [recentVideo, setRecentVideo]     = useState(null);

  useEffect(() => {
    setBookmarkCount(getBookmarks().length);
    const lib = getLibrary();
    if (lib.length > 0) setRecentVideo(lib[0]);
  }, [active]); // re-read when active screen changes (user may have added items)

  const items = [
    { key: "home",      label: "Home"      },
    { key: "library",   label: "Library"   },
    { key: "bookmarks", label: "Bookmarks", count: bookmarkCount },
    { key: "settings",  label: "Settings"  },
  ];

  return (
    <aside className="w-[200px] bg-paper-warm border-r border-paper-edge px-5 py-7 flex flex-col shrink-0">
      {/* Wordmark block */}
      <div className="pb-8 border-b border-paper-rule">
        <div className="font-serif text-[22px] text-ink leading-none">
          Synapse
        </div>
      </div>

      {/* Nav */}
      <nav className="mt-6 flex flex-col font-sans text-[13px]">
        {items.map((item) => {
          const isActive = item.key === active;
          return (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              className={
                "flex items-center gap-2.5 text-left transition-colors " +
                (isActive
                  ? "py-2 text-ink"
                  : "py-2 px-3.5 text-ink-faded hover:text-ink")
              }
            >
              {isActive && (
                <span className="block w-1 h-4 bg-accent rounded-[1px]" />
              )}
              <span className="flex-1">{item.label}</span>
              {item.count > 0 && (
                <span className="font-mono text-[10px] text-ink-ghost bg-paper-edge px-1.5 py-0.5 rounded-full">
                  {item.count}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer block — recently processed */}
      <div className="mt-auto font-sans">
        <div className="eyebrow text-eyebrow-sm mb-2">Recently processed</div>
        {recentVideo ? (
          <div className="text-[12px] text-ink-muted leading-snug">
            <div className="truncate">{recentVideo.title || recentVideo.videoTitle}</div>
            <div className="text-ink-ghost">
              {new Date(recentVideo.savedAt || recentVideo.processedAt).toLocaleDateString("en-US", {
                month: "short", day: "numeric",
              })}
            </div>
          </div>
        ) : (
          <div className="text-[12px] text-ink-ghost leading-snug">
            No videos yet.
          </div>
        )}
      </div>
    </aside>
  );
}
