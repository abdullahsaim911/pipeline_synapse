const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("synapseProtocol", {
  getDataUrl: (filePath) => {
    if (!filePath) return "";
    const normalizedPath = filePath.replace(/\\/g, "/");
    return `synapse://files/${normalizedPath}`;
  },
  exportAudio: (sourcePath, fileName) =>
    ipcRenderer.invoke("export-audio", { sourcePath, fileName }),
});

console.log("[Preload] synapseProtocol exposed to window");
