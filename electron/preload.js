const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("traceDesktop", {
  notifyReady() {
    ipcRenderer.send("trace:renderer-ready");
  },
  getSplashState() {
    return ipcRenderer.invoke("trace:get-splash-state");
  },
  onSplashStatus(callback) {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("trace:splash-status", listener);
    return () => {
      ipcRenderer.removeListener("trace:splash-status", listener);
    };
  },
  getWelcomeModel() {
    return ipcRenderer.invoke("trace:get-welcome-model");
  },
  launchWorkspace() {
    return ipcRenderer.invoke("trace:launch-workspace");
  },
  quitApp() {
    return ipcRenderer.invoke("trace:quit-app");
  },
});
