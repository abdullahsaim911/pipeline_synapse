const { app, BrowserWindow, protocol, net } = require("electron");
const path = require("path");
const fs = require("fs");

const isDev = !app.isPackaged;

const MIME_TYPES = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".m4a": "audio/mp4",
  ".ogg": "audio/ogg",
};

function registerCustomProtocol() {
  protocol.handle("synapse", (request) => {
    const url = new URL(request.url);
    const filePath = url.pathname;

    console.log("[Protocol] Requested:", url.href);
    console.log("[Protocol] Pathname:", filePath);

    // URL format is synapse://files/data/video_id/keyframes/...
    // The pathname should be something like /data/P_SD5Rt6XMk/keyframes/...
    const fullPath = path.join(path.dirname(__dirname), filePath);

    console.log("[Protocol] Resolved path:", fullPath);

    if (!fs.existsSync(fullPath)) {
      console.log("[Protocol] File not found:", fullPath);
      return new Response("File Not Found", { status: 404 });
    }

    try {
      const ext = path.extname(fullPath).toLowerCase();
      const mimeType = MIME_TYPES[ext] || "application/octet-stream";
      const data = fs.readFileSync(fullPath);

      console.log("[Protocol] Serving file:", fullPath, `(${mimeType})`);
      return new Response(data, {
        headers: {
          "Content-Type": mimeType,
          "Content-Length": data.length.toString(),
        },
      });
    } catch (err) {
      console.error("[Protocol] Error serving file:", err);
      return new Response("Internal Server Error", { status: 500 });
    }
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#F7F4EC",
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, "src", "preload.js"),
    },
  });

  if (isDev) {
    win.loadURL("http://localhost:5173");
  } else {
    win.loadFile(path.join(__dirname, "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  registerCustomProtocol();
  createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
