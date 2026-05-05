const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("synapseProtocol", {
  getDataUrl: (filePath) => {
    if (!filePath) return "";
    const normalizedPath = filePath.replace(/\\/g, "/");
    // Keep the "data/" prefix in the pathname, use "files" as hostname
    return `synapse://files/${normalizedPath}`;
  },
});

console.log("[Preload] synapseProtocol exposed to window");
