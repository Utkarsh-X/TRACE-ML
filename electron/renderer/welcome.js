const greetingEl = document.getElementById("greeting");
const subtitleEl = document.getElementById("subtitle");
const detailEl = document.getElementById("detail");
const serviceUrlEl = document.getElementById("service-url");
const dataRootEl = document.getElementById("data-root");
const systemLabelEl = document.getElementById("system-label");
const launchButton = document.getElementById("launch-button");
const skipButton = document.getElementById("skip-button");

async function initWelcome() {
  const model = await window.traceDesktop.getWelcomeModel();
  greetingEl.textContent = model.greeting;
  subtitleEl.textContent = model.subtitle;
  detailEl.textContent = model.detail;
  serviceUrlEl.textContent = model.serviceUrl;
  dataRootEl.textContent = model.dataRoot;
  systemLabelEl.textContent = model.systemLabel;
  launchButton.textContent = model.ctaLabel;
  skipButton.textContent = model.skipLabel;
}

async function launchWorkspace() {
  launchButton.disabled = true;
  skipButton.disabled = true;
  launchButton.textContent = "Opening Workspace…";
  await window.traceDesktop.launchWorkspace();
}

launchButton.addEventListener("click", launchWorkspace);
skipButton.addEventListener("click", launchWorkspace);
window.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    launchWorkspace();
  }
});

initWelcome();
