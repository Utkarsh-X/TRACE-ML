/**
 * Enrollment Page Controller
 *
 * - Lists registered persons (left sidebar)
 * - Person creation form + image upload (center)
 * - Training trigger + status polling (right sidebar)
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  var _selectedFiles = [];
  var _trainPollTimer = null;

  /* ─── Person list (left sidebar) ─── */

  function loadPersonList() {
    TraceClient.listPersons().then(function (persons) {
      if (!persons) return;
      var root = $("person-list-root");
      var countEl = $("person-count");
      if (countEl) countEl.textContent = String(persons.length);
      if (!root) return;

      if (persons.length === 0) {
        root.innerHTML = '<div class="text-center py-8 text-outline text-[0.75rem]">No persons registered yet</div>';
        return;
      }

      root.innerHTML = persons.map(function (p) {
        var stateClass = "lc-" + (p.lifecycle_state || "draft");
        var stateLabel = (p.lifecycle_state || "draft").toUpperCase();
        return '<div class="person-card" data-person-id="' + TraceClient.escapeHtml(p.person_id) + '">'
          + '<div class="flex items-center justify-between mb-0.5">'
          + '<span class="font-headline font-semibold text-[0.8rem] text-primary">' + TraceClient.escapeHtml(p.person_id) + '</span>'
          + '<span class="font-mono text-[0.55rem] ' + stateClass + '">' + stateLabel + '</span>'
          + '</div>'
          + '<div class="text-[0.75rem] text-on-surface">' + TraceClient.escapeHtml(p.name) + '</div>'
          + '<div class="flex items-center gap-2 mt-0.5">'
          + '<span class="font-mono text-[0.6rem] text-outline">' + TraceClient.escapeHtml(p.category) + '</span>'
          + '<span class="font-mono text-[0.6rem] text-outline">imgs: ' + (p.image_count_on_disk || 0) + '</span>'
          + '<span class="font-mono text-[0.6rem] text-outline">emb: ' + (p.valid_embeddings || 0) + '</span>'
          + '</div>'
          + '</div>';
      }).join("");

      // Wire click handlers for person detail
      root.querySelectorAll(".person-card").forEach(function (card) {
        card.addEventListener("click", function () {
          root.querySelectorAll(".person-card").forEach(function (c) { c.classList.remove("active"); });
          card.classList.add("active");
          loadPersonDetail(card.getAttribute("data-person-id"));
        });
      });
    });
  }

  function loadPersonDetail(personId) {
    TraceClient.getPerson(personId).then(function (p) {
      if (!p) return;
      var panel = $("person-detail-panel");
      if (!panel) return;
      var score = (p.enrollment_score || 0).toFixed(2);
      var stateClass = "lc-" + (p.lifecycle_state || "draft");
      panel.innerHTML =
        '<div class="flex justify-between"><span class="stat-label">ID</span>'
        + '<span class="font-mono text-[0.7rem] text-primary">' + TraceClient.escapeHtml(p.person_id) + '</span></div>'
        + '<div class="flex justify-between"><span class="stat-label">Name</span>'
        + '<span class="text-[0.7rem]">' + TraceClient.escapeHtml(p.name) + '</span></div>'
        + '<div class="flex justify-between"><span class="stat-label">Category</span>'
        + '<span class="text-[0.7rem]">' + TraceClient.escapeHtml(p.category) + '</span></div>'
        + '<div class="flex justify-between"><span class="stat-label">State</span>'
        + '<span class="font-mono text-[0.7rem] ' + stateClass + '">' + (p.lifecycle_state || "draft").toUpperCase() + '</span></div>'
        + '<div class="flex justify-between"><span class="stat-label">Score</span>'
        + '<span class="font-mono text-[0.7rem]">' + score + '</span></div>'
        + '<div class="flex justify-between"><span class="stat-label">Images</span>'
        + '<span class="font-mono text-[0.7rem]">' + (p.image_count_on_disk || 0) + '</span></div>'
        + '<div class="flex justify-between"><span class="stat-label">Embeddings</span>'
        + '<span class="font-mono text-[0.7rem]">' + (p.valid_embeddings || 0) + '</span></div>'
        + '<div class="flex justify-between"><span class="stat-label">Reason</span>'
        + '<span class="font-mono text-[0.6rem] text-outline">' + TraceClient.escapeHtml(p.lifecycle_reason || "—") + '</span></div>'
        + '<button type="button" class="btn-secondary w-full mt-3" id="btn-delete-person" '
        + 'data-pid="' + TraceClient.escapeHtml(p.person_id) + '">Delete Person</button>';

      // Wire delete
      var delBtn = $("btn-delete-person");
      if (delBtn) {
        delBtn.addEventListener("click", function () {
          var pid = delBtn.getAttribute("data-pid");
          if (!confirm("Delete " + pid + " and all linked data?")) return;
          TraceClient.deletePerson(pid).then(function () {
            loadPersonList();
            panel.innerHTML = '<p class="text-[0.75rem] text-outline text-center py-4">Person deleted</p>';
          });
        });
      }
    });
  }

  /* ─── Image upload & drag-drop ─── */

  function initDropZone() {
    var zone = $("drop-zone");
    var fileInput = $("file-input");
    if (!zone || !fileInput) return;

    zone.addEventListener("click", function () { fileInput.click(); });

    zone.addEventListener("dragover", function (e) {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", function () {
      zone.classList.remove("drag-over");
    });
    zone.addEventListener("drop", function (e) {
      e.preventDefault();
      zone.classList.remove("drag-over");
      addFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener("change", function () {
      addFiles(fileInput.files);
      fileInput.value = "";
    });
  }

  function addFiles(fileList) {
    for (var i = 0; i < fileList.length; i++) {
      var f = fileList[i];
      if (f.type && f.type.startsWith("image/")) {
        _selectedFiles.push(f);
      }
    }
    renderThumbnails();
    updateCreateButton();
  }

  function renderThumbnails() {
    var grid = $("thumb-preview");
    var countEl = $("file-count");
    if (countEl) countEl.textContent = _selectedFiles.length + " images selected";
    if (!grid) return;

    grid.innerHTML = "";
    _selectedFiles.forEach(function (file, idx) {
      var img = document.createElement("img");
      img.alt = file.name;
      img.title = file.name;
      var url = URL.createObjectURL(file);
      img.src = url;
      img.addEventListener("click", function () {
        _selectedFiles.splice(idx, 1);
        renderThumbnails();
        updateCreateButton();
      });
      img.style.cursor = "pointer";
      img.title = "Click to remove: " + file.name;
      grid.appendChild(img);
    });
  }

  /* ─── Form logic ─── */

  function updateCreateButton() {
    var btn = $("btn-create-upload");
    var name = ($("enroll-name") || {}).value || "";
    if (btn) btn.disabled = !(name.trim().length > 0 && _selectedFiles.length > 0);
  }

  function clearForm() {
    var ids = ["enroll-name", "enroll-dob", "enroll-city", "enroll-country", "enroll-notes"];
    ids.forEach(function (id) { var el = $(id); if (el) el.value = ""; });
    var cat = $("enroll-category"); if (cat) cat.selectedIndex = 0;
    var sev = $("enroll-severity"); if (sev) sev.selectedIndex = 0;
    var gen = $("enroll-gender"); if (gen) gen.selectedIndex = 0;
    _selectedFiles = [];
    renderThumbnails();
    updateCreateButton();
    var status = $("enroll-status"); if (status) status.textContent = "";
  }

  function handleCreateUpload() {
    var btn = $("btn-create-upload");
    var statusEl = $("enroll-status");
    if (btn) btn.disabled = true;
    if (statusEl) statusEl.textContent = "Creating person...";

    var payload = {
      name: ($("enroll-name") || {}).value || "",
      category: ($("enroll-category") || {}).value || "criminal",
      severity: ($("enroll-severity") || {}).value || "",
      dob: ($("enroll-dob") || {}).value || "",
      gender: ($("enroll-gender") || {}).value || "",
      city: ($("enroll-city") || {}).value || "",
      country: ($("enroll-country") || {}).value || "",
      notes: ($("enroll-notes") || {}).value || "",
    };

    TraceClient.createPerson(payload).then(function (result) {
      if (!result || !result.person_id) {
        if (statusEl) statusEl.textContent = "Failed to create person";
        if (btn) btn.disabled = false;
        return;
      }
      if (statusEl) statusEl.textContent = "Created " + result.person_id + ". Uploading images...";

      // Upload images
      TraceClient.uploadPersonImages(result.person_id, _selectedFiles).then(function (uploadResult) {
        if (uploadResult) {
          if (statusEl) statusEl.textContent = "Done! " + result.person_id + " — " + uploadResult.uploaded + " images uploaded.";
        } else {
          if (statusEl) statusEl.textContent = "Person created but image upload failed.";
        }
        clearForm();
        loadPersonList();
      });
    }).catch(function () {
      if (statusEl) statusEl.textContent = "Error creating person";
      if (btn) btn.disabled = false;
    });
  }

  /* ─── Training ─── */

  function handleTrainRebuild() {
    var btn = $("btn-train-rebuild");
    var runningEl = $("train-running");
    if (btn) btn.disabled = true;
    if (runningEl) runningEl.textContent = "Starting...";

    TraceClient.trainRebuild({ scope: "all" }).then(function (result) {
      if (result && result.status === "started") {
        if (runningEl) runningEl.textContent = "Running...";
        startTrainPoll();
      } else if (result && result.status === "already_running") {
        if (runningEl) runningEl.textContent = "Already running...";
        startTrainPoll();
      } else {
        if (runningEl) runningEl.textContent = "Failed to start";
        if (btn) btn.disabled = false;
      }
    });
  }

  function startTrainPoll() {
    if (_trainPollTimer) return;
    _trainPollTimer = setInterval(pollTrainStatus, 2000);
  }

  function pollTrainStatus() {
    TraceClient.trainStatus().then(function (status) {
      if (!status) return;
      var runningEl = $("train-running");
      var lastRunEl = $("train-last-run");
      var embEl = $("train-embeddings");
      var activeEl = $("train-active");
      var readyEl = $("train-ready");
      var blockedEl = $("train-blocked");
      var btn = $("btn-train-rebuild");

      if (status.running) {
        if (runningEl) runningEl.textContent = "Running...";
      } else {
        if (runningEl) runningEl.textContent = "Idle";
        if (btn) btn.disabled = false;
        if (_trainPollTimer) {
          clearInterval(_trainPollTimer);
          _trainPollTimer = null;
        }
        // Refresh person list after training completes
        loadPersonList();
      }

      if (status.last_completed_at) {
        if (lastRunEl) lastRunEl.textContent = TraceClient.formatDateTime(status.last_completed_at);
      }

      var r = status.last_result;
      if (r && !r.error) {
        if (embEl) embEl.textContent = String(r.embeddings_created || 0);
        if (activeEl) activeEl.textContent = String(r.active_persons || 0);
        if (readyEl) readyEl.textContent = String(r.ready_persons || 0);
        if (blockedEl) blockedEl.textContent = String(r.blocked_persons || 0);
      } else if (r && r.error) {
        if (embEl) embEl.textContent = "Error";
      }
    });
  }


  /* --- Camera Capture Controller ---------------------------------------- */

  var _camStream = null;
  var _camDevices = [];
  var _camDeviceIndex = 0;
  var _camCaptures = []; // Array of {blob, dataUrl}
  var _camTimerEnabled = false;
  var _camCountdownTimer = null;

  function openCameraModal() {
    var modal = document.getElementById("camera-modal");
    if (!modal) return;
    modal.classList.add("open");
    _camCaptures = [];
    renderCapturedBar();
    startCameraStream();
  }

  function closeCameraModal() {
    var modal = document.getElementById("camera-modal");
    if (!modal) return;
    modal.classList.remove("open");
    stopCameraStream();
    if (_camCountdownTimer) { clearTimeout(_camCountdownTimer); _camCountdownTimer = null; }
    var cd = document.getElementById("cam-countdown");
    if (cd) { cd.textContent = ""; cd.classList.remove("show"); }
    var snapBtn = document.getElementById("cam-snap-btn");
    if (snapBtn) snapBtn.disabled = false;
  }

  function stopCameraStream() {
    if (_camStream) {
      _camStream.getTracks().forEach(function (t) { t.stop(); });
      _camStream = null;
    }
    var video = document.getElementById("cam-video");
    if (video) { video.srcObject = null; }
    var recDot = document.getElementById("cam-rec-dot");
    if (recDot) recDot.classList.remove("active");
    setCamOverlay(true, "videocam_off", "Camera stopped");
  }

  function setCamOverlay(visible, icon, text) {
    var overlay = document.getElementById("cam-status-overlay");
    var textEl = document.getElementById("cam-status-text");
    if (!overlay) return;
    overlay.style.display = visible ? "flex" : "none";
    if (icon) {
      var iconEl = overlay.querySelector(".cam-status-icon");
      if (iconEl) iconEl.textContent = icon;
    }
    if (text && textEl) textEl.textContent = text;
  }

  function enumerateCameraDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      _camDevices = [];
      return Promise.resolve([]);
    }
    return navigator.mediaDevices.enumerateDevices().then(function (devices) {
      _camDevices = devices.filter(function (d) { return d.kind === "videoinput"; });
      var switchBtn = document.getElementById("cam-switch-btn");
      if (switchBtn) switchBtn.disabled = _camDevices.length < 2;
      return _camDevices;
    });
  }

  function startCameraStream(deviceId) {
    setCamOverlay(true, "hourglass_top", "Requesting camera access...");
    var snapBtn = document.getElementById("cam-snap-btn");
    if (snapBtn) snapBtn.disabled = true;
    if (_camStream) { stopCameraStream(); }

    var constraints = {
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
      .then(function (stream) {
        _camStream = stream;
        var video = document.getElementById("cam-video");
        if (video) {
          video.srcObject = stream;
          video.onloadedmetadata = function () {
            video.play();
            setCamOverlay(false);
            if (snapBtn) snapBtn.disabled = false;
            var recDot = document.getElementById("cam-rec-dot");
            if (recDot) recDot.classList.add("active");
          };
        }
        enumerateCameraDevices();
      })
      .catch(function (err) {
        console.warn("Camera access error:", err);
        var msg = "Camera access denied";
        if (err && err.name === "NotFoundError") msg = "No camera found";
        if (err && err.name === "NotAllowedError") msg = "Permission denied - allow camera in browser";
        setCamOverlay(true, "videocam_off", msg);
        if (snapBtn) snapBtn.disabled = true;
      });
  }

  function switchCamera() {
    if (_camDevices.length < 2) return;
    _camDeviceIndex = (_camDeviceIndex + 1) % _camDevices.length;
    startCameraStream(_camDevices[_camDeviceIndex].deviceId);
  }

  function doSnapNow() {
    var video = document.getElementById("cam-video");
    var canvas = document.getElementById("cam-canvas");
    if (!video || !canvas || !_camStream) return;

    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    var ctx = canvas.getContext("2d");

    // Draw mirrored to match the CSS mirror on the video
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();

    triggerFlash();

    canvas.toBlob(function (blob) {
      if (!blob) return;
      var reader = new FileReader();
      reader.onload = function (e) {
        _camCaptures.push({ blob: blob, dataUrl: e.target.result });
        renderCapturedBar();
      };
      reader.readAsDataURL(blob);
    }, "image/jpeg", 0.92);
  }

  function triggerFlash() {
    var flash = document.getElementById("cam-flash");
    if (!flash) return;
    flash.classList.remove("fade");
    flash.classList.add("bang");
    setTimeout(function () {
      flash.classList.remove("bang");
      flash.classList.add("fade");
    }, 80);
    setTimeout(function () { flash.classList.remove("fade"); }, 500);
  }

  function snapWithCountdown() {
    var snapBtn = document.getElementById("cam-snap-btn");
    var cd = document.getElementById("cam-countdown");
    if (!cd) { doSnapNow(); return; }
    if (snapBtn) snapBtn.disabled = true;

    var count = 3;
    cd.textContent = String(count);
    cd.classList.add("show");

    function tick() {
      count--;
      if (count > 0) {
        cd.textContent = String(count);
        _camCountdownTimer = setTimeout(tick, 1000);
      } else {
        cd.textContent = "";
        cd.classList.remove("show");
        _camCountdownTimer = null;
        doSnapNow();
        if (snapBtn) snapBtn.disabled = false;
      }
    }
    _camCountdownTimer = setTimeout(tick, 1000);
  }

  function handleSnap() {
    if (!_camStream) return;
    if (_camTimerEnabled) { snapWithCountdown(); } else { doSnapNow(); }
  }

  function renderCapturedBar() {
    var bar = document.getElementById("cam-captured-bar");
    var emptyEl = document.getElementById("cam-captured-empty");
    var countEl = document.getElementById("cam-captured-count");
    var addBtn = document.getElementById("cam-add-btn");
    if (!bar) return;

    if (countEl) countEl.textContent = _camCaptures.length + " captured";
    if (addBtn) addBtn.disabled = _camCaptures.length === 0;

    var existing = bar.querySelectorAll(".cam-captured-thumb");
    existing.forEach(function (el) { el.remove(); });

    if (_camCaptures.length === 0) {
      if (emptyEl) emptyEl.style.display = "";
      return;
    }
    if (emptyEl) emptyEl.style.display = "none";

    _camCaptures.forEach(function (cap, idx) {
      var thumb = document.createElement("div");
      thumb.className = "cam-captured-thumb";
      thumb.title = "Click to remove capture";
      thumb.innerHTML = '<img src="' + cap.dataUrl + '" alt="Capture ' + (idx + 1) + '" />'
        + '<div class="cam-remove-badge"><span class="material-symbols-outlined" style="font-size:20px">delete</span></div>';
      thumb.addEventListener("click", function () {
        _camCaptures.splice(idx, 1);
        renderCapturedBar();
      });
      bar.appendChild(thumb);
    });
  }

  function addCapturesToUpload() {
    if (_camCaptures.length === 0) return;
    _camCaptures.forEach(function (cap, idx) {
      var filename = "camera_" + Date.now() + "_" + idx + ".jpg";
      var file = new File([cap.blob], filename, { type: "image/jpeg" });
      _selectedFiles.push(file);
    });
    renderThumbnails();
    updateCreateButton();
    closeCameraModal();
  }

  function initCameraModal() {
    var openBtn = document.getElementById("btn-open-camera");
    if (openBtn) openBtn.addEventListener("click", openCameraModal);

    var closeBtn = document.getElementById("cam-close-btn");
    if (closeBtn) closeBtn.addEventListener("click", closeCameraModal);

    var modal = document.getElementById("camera-modal");
    if (modal) {
      modal.addEventListener("click", function (e) {
        if (e.target === modal) closeCameraModal();
      });
    }

    var snapBtn = document.getElementById("cam-snap-btn");
    if (snapBtn) snapBtn.addEventListener("click", handleSnap);

    var timerBtn = document.getElementById("cam-timer-btn");
    if (timerBtn) {
      timerBtn.addEventListener("click", function () {
        _camTimerEnabled = !_camTimerEnabled;
        timerBtn.classList.toggle("active", _camTimerEnabled);
      });
    }

    var switchBtn = document.getElementById("cam-switch-btn");
    if (switchBtn) switchBtn.addEventListener("click", switchCamera);

    var discardBtn = document.getElementById("cam-discard-btn");
    if (discardBtn) {
      discardBtn.addEventListener("click", function () {
        _camCaptures = [];
        renderCapturedBar();
      });
    }

    var addBtn = document.getElementById("cam-add-btn");
    if (addBtn) addBtn.addEventListener("click", addCapturesToUpload);

    document.addEventListener("keydown", function (e) {
      var m = document.getElementById("camera-modal");
      if (e.key === "Escape" && m && m.classList.contains("open")) {
        closeCameraModal();
      }
    });
  }
  /* ─── Init ─── */

  function init() {
    var mainContent = document.querySelector("main");
    if (typeof TraceRender !== "undefined" && TraceRender.initOfflineUI) {
      TraceRender.initOfflineUI(mainContent);
    }

    initDropZone();
    initCameraModal();

    // Wire form events
    var nameInput = $("enroll-name");
    if (nameInput) nameInput.addEventListener("input", updateCreateButton);

    var createBtn = $("btn-create-upload");
    if (createBtn) createBtn.addEventListener("click", handleCreateUpload);

    var clearBtn = $("btn-clear-form");
    if (clearBtn) clearBtn.addEventListener("click", clearForm);

    var trainBtn = $("btn-train-rebuild");
    if (trainBtn) trainBtn.addEventListener("click", handleTrainRebuild);

    // Initial load
    TraceClient.probe().then(function (info) {
      if (info) {
        loadPersonList();
        pollTrainStatus(); // Show last training result
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
