// localStorage keys
const BOOKMARKS_KEY = "synapse_bookmarks";
const LIBRARY_KEY   = "synapse_library";

/* ── Bookmarks ───────────────────────────────────────────────────────────── */

export function getBookmarks() {
  try { return JSON.parse(localStorage.getItem(BOOKMARKS_KEY) || "[]"); }
  catch { return []; }
}

export function saveBookmark(bookmark) {
  const list = getBookmarks().filter((b) => b.id !== bookmark.id);
  list.unshift(bookmark);
  localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(list));
}

export function removeBookmark(id) {
  const list = getBookmarks().filter((b) => b.id !== id);
  localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(list));
}

export function isBookmarked(id) {
  return getBookmarks().some((b) => b.id === id);
}

/* ── Library ─────────────────────────────────────────────────────────────── */

export function getLibrary() {
  try { return JSON.parse(localStorage.getItem(LIBRARY_KEY) || "[]"); }
  catch { return []; }
}

export function saveToLibrary(item) {
  const list = getLibrary().filter((e) => e.id !== item.id);
  list.unshift(item);
  localStorage.setItem(LIBRARY_KEY, JSON.stringify(list));
}

export function removeFromLibrary(id) {
  const list = getLibrary().filter((e) => e.id !== id);
  localStorage.setItem(LIBRARY_KEY, JSON.stringify(list));
}
