import { useState, useEffect } from "react";
import Home from "./components/Home";
import Processing from "./components/Processing";
import Timeline from "./components/Timeline";
import Explanation from "./components/Explanation";
import BookmarksScreen from "./components/BookmarksScreen";
import LibraryScreen from "./components/LibraryScreen";
import Navbar from "./components/Navbar";
import { processVideo, resetMockTimer } from "./api";

function App() {
  const [screen,  setScreen]  = useState("home");
  const [jobInfo, setJobInfo] = useState(null);

  // ── Theme ──────────────────────────────────────────────────────────────────
  const [theme, setTheme] = useState(
    () => localStorage.getItem("synapse_theme") || "paper"
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("synapse_theme", theme);
  }, [theme]);

  // ── HOME -> PROCESSING ─────────────────────────────────────────────────────
  const handleProcess = ({ url, videoId, mode }) => {
    resetMockTimer();
    const promise = processVideo({ url, videoId, mode });
    setJobInfo({ jobId: videoId, videoId, mode, promise });
    setScreen("processing");
  };

  // ── PROCESSING -> TIMELINE ─────────────────────────────────────────────────
  const handleProcessingDone = (jobId) => {
    setJobInfo((prev) => ({ ...prev, jobId, promise: undefined }));
    setScreen("timeline");
  };

  const handleProcessingCancel = () => {
    setJobInfo(null);
    setScreen("home");
  };

  // ── TIMELINE -> EXPLANATION ────────────────────────────────────────────────
  const handleOpenPoint = (point) => {
    setJobInfo({ ...jobInfo, pointId: point.id });
    setScreen("explanation");
  };

  const handleBackToTimeline = () => {
    if (jobInfo?.bookmarkData) setScreen("bookmarks");
    else if (jobInfo?.autoPlay)  setScreen("library");
    else                         setScreen("timeline");
  };

  // ── BOOKMARKS -> EXPLANATION ───────────────────────────────────────────────
  const handleOpenBookmark = (bookmark) => {
    setJobInfo({
      jobId:        bookmark.jobId,
      videoId:      bookmark.videoId,
      pointId:      bookmark.pointId,
      mode:         bookmark.mode,
      bookmarkData: bookmark,
    });
    setScreen("explanation");
  };

  // ── LIBRARY -> EXPLANATION ─────────────────────────────────────────────────
  const handleOpenLibraryExplanation = ({ pointId, videoId, jobId, mode, autoPlay }) => {
    setJobInfo({ jobId, videoId, pointId, mode, autoPlay: !!autoPlay });
    setScreen("explanation");
  };

  // ── SIDEBAR NAVIGATION ─────────────────────────────────────────────────────
  const handleNavigate = (key) => {
    if (key === "home") {
      setJobInfo(null);
      setScreen("home");
    } else if (key === "bookmarks" || key === "library") {
      setScreen(key);
    }
  };

  // ── Active screen ──────────────────────────────────────────────────────────
  let activeScreen;

  if (screen === "processing" && jobInfo) {
    activeScreen = (
      <Processing
        jobId={jobInfo.jobId}
        videoId={jobInfo.videoId}
        mode={jobInfo.mode}
        promise={jobInfo.promise}
        onDone={handleProcessingDone}
        onCancel={handleProcessingCancel}
        onNavigate={handleNavigate}
      />
    );
  } else if (screen === "timeline" && jobInfo) {
    activeScreen = (
      <Timeline
        jobId={jobInfo.jobId}
        videoId={jobInfo.videoId}
        mode={jobInfo.mode}
        onOpenPoint={handleOpenPoint}
        onNavigate={handleNavigate}
      />
    );
  } else if (screen === "explanation" && jobInfo) {
    activeScreen = (
      <Explanation
        jobId={jobInfo.jobId}
        videoId={jobInfo.videoId}
        pointId={jobInfo.pointId}
        mode={jobInfo.mode}
        bookmarkData={jobInfo.bookmarkData || null}
        autoPlay={jobInfo.autoPlay || false}
        onBack={handleBackToTimeline}
        onNavigate={handleNavigate}
      />
    );
  } else if (screen === "bookmarks") {
    activeScreen = (
      <BookmarksScreen
        onNavigate={handleNavigate}
        onOpenBookmark={handleOpenBookmark}
      />
    );
  } else if (screen === "library") {
    activeScreen = (
      <LibraryScreen
        onNavigate={handleNavigate}
        onOpenExplanation={handleOpenLibraryExplanation}
      />
    );
  } else {
    activeScreen = <Home onProcess={handleProcess} onNavigate={handleNavigate} />;
  }

  // ── Layout: Navbar (slim top bar) + screen below ───────────────────────────
  return (
    <div className="flex flex-col h-screen">
      <Navbar theme={theme} onThemeChange={setTheme} />
      <div className="flex-1 overflow-hidden">
        {activeScreen}
      </div>
    </div>
  );
}

export default App;
