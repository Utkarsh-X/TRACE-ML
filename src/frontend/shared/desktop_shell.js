(function () {
  "use strict";

  function bindDesktopExit() {
    var exitButton = document.getElementById("desktop-exit-button");
    if (!exitButton) {
      return;
    }

    exitButton.addEventListener("click", function () {
      if (window.traceDesktop && typeof window.traceDesktop.quitApp === "function") {
        window.traceDesktop.quitApp();
        return;
      }

      window.close();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindDesktopExit, { once: true });
  } else {
    bindDesktopExit();
  }
})();
