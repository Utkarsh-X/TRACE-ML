/**
 * Enrollment Page Controller
 *
 * - Lists registered persons (left sidebar)
 * - Person creation form + image upload (center)
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  var _allPersons = [];
  var _selectedFiles = [];

  /* ─── Person list (left sidebar) ─── */

  function loadPersonList() {
    TraceClient.listPersons().then(function (persons) {
      if (!persons) return;
      _allPersons = persons;
      renderPersonList(persons);
    });
  }

  function renderPersonList(persons) {
    var root = $("person-list-root");
    var countEl = $("person-count");
    if (countEl) countEl.textContent = String(persons.length);
    if (!root) return;

    if (persons.length === 0) {
      root.innerHTML = '<div class="text-center py-12 text-outline text-[0.7rem] font-mono">NO ENTITIES REGISTERED</div>';
      return;
    }

    root.innerHTML = persons.map(function (p) {
      var stateClass = "lc-" + (p.lifecycle_state || "draft");
      var stateLabel = (p.lifecycle_state || "draft").toUpperCase();
      var cat = (p.category || "UNKNOWN").toUpperCase();
      
      return '<div class="person-card" data-person-id="' + TraceClient.escapeHtml(p.person_id) + '">'
        + '<div class="card-avatar-frame">'
        + '<div class="avatar-corner corner-tl"></div><div class="avatar-corner corner-tr"></div>'
        + '<div class="avatar-corner corner-bl"></div><div class="avatar-corner corner-br"></div>'
        + '<span class="material-symbols-outlined">person</span>'
        + '</div>'
        + '<div class="card-content">'
        + '<div class="card-id">' + TraceClient.escapeHtml(p.person_id) + '</div>'
        + '<div class="card-name">' + TraceClient.escapeHtml(p.name) + '</div>'
        + '<div class="card-meta">'
        + '<span class="card-badge ' + stateClass + '">' + stateLabel + '</span>'
        + '<div class="card-metric">'
        + '<span class="material-symbols-outlined">image</span>'
        + '<span>' + (p.image_count_on_disk || 0) + '</span>'
        + '</div>'
        + '<div class="card-metric">'
        + '<span class="material-symbols-outlined">analytics</span>'
        + '<span>' + (p.valid_embeddings || 0) + '</span>'
        + '</div>'
        + '</div>'
        + '</div>'
        + '</div>';
    }).join("");

    // Wire click handlers
    root.querySelectorAll(".person-card").forEach(function (card) {
      card.addEventListener("click", function () {
        root.querySelectorAll(".person-card").forEach(function (c) { c.classList.remove("active"); });
        card.classList.add("active");
        loadPersonDetail(card.getAttribute("data-person-id"));
      });
    });
  }

  function wireSearch() {
    var search = $("person-search");
    if (!search) return;
    search.addEventListener("input", function() {
      var q = search.value.toLowerCase().trim();
      if (!q) {
        renderPersonList(_allPersons);
        return;
      }
      var filtered = _allPersons.filter(function(p) {
        return p.name.toLowerCase().indexOf(q) >= 0 || p.person_id.toLowerCase().indexOf(q) >= 0;
      });
      renderPersonList(filtered);
    });
  }

  function loadPersonDetail(personId) {
    // Auto-open the right intelligence panel whenever a record is selected.
    var rightPanel = $("enroll-right");
    var center = $("enroll-center");
    if (rightPanel) rightPanel.classList.remove("collapsed");
    if (center) center.classList.remove("intel-hidden");

    TraceClient.getPerson(personId).then(function (p) {
      if (!p) return;
      var panel = $("person-detail-panel");
      if (!panel) return;

      // Hide the empty-state placeholder now that we have real content.
      var emptyState = $("intel-empty-state");
      if (emptyState) emptyState.style.display = "none";

      var score = (p.enrollment_score || 0);
      var scoreColor = score > 0.8 ? "text-primary" : (score > 0.5 ? "text-warning" : "text-error");
      var stateClass = "lc-" + (p.lifecycle_state || "draft");
      
      panel.innerHTML = '<div class="space-y-6">'
        + '<div>'
        + '<div class="flex items-center justify-between mb-2">'
        + '<span class="font-mono text-[0.6rem] text-outline uppercase">Biometric Quality</span>'
        + '<span class="font-mono text-[0.8rem] font-bold ' + scoreColor + '">' + (score * 100).toFixed(0) + '%</span>'
        + '</div>'
        + '<div class="h-1 bg-surface-high w-full rounded-full overflow-hidden">'
        + '<div class="h-full bg-current transition-all duration-500 ' + scoreColor + '" style="width:' + (score * 100) + '%"></div>'
        + '</div>'
        + '</div>'
        
        + '<div class="space-y-2.5">'
        + detailRow("System ID", p.person_id, "text-primary font-bold font-mono")
        + detailRow("Lifecycle", (p.lifecycle_state || "DRAFT").toUpperCase(), stateClass + " font-mono font-bold")
        + detailRow("Enrollment", p.enrollment_status || "PENDING", "uppercase font-mono")
        + detailRow("Record Created", TraceClient.formatDateTime(p.created_at), "text-outline font-mono")
        + '</div>'
        
        + '<div class="pt-4 border-t border-outline-variant/10">'
        + '<h4 class="font-mono text-[0.6rem] text-outline uppercase mb-3">Intelligence Metadata</h4>'
        + '<div class="space-y-2.5">'
        + detailRow("Priority", (p.severity || "NORMAL").toUpperCase())
        + detailRow("Gender", (p.gender || "UNKNOWN").toUpperCase())
        + detailRow("D.O.B", p.dob || "UNKNOWN")
        + detailRow("Origin", (p.city || "") + (p.city && p.country ? ", " : "") + (p.country || "UNKNOWN"))
        + '</div>'
        + '</div>'
        
        + '<div class="pt-4 border-t border-outline-variant/10">'
        + '<h4 class="font-mono text-[0.6rem] text-outline uppercase mb-2">Internal Notes</h4>'
        + '<p class="text-[0.7rem] text-outline-variant leading-relaxed">' + TraceClient.escapeHtml(p.notes || "No intelligence notes recorded for this entity.") + '</p>'
        + '</div>'
        
        + '<div class="pt-6">'
        + '<button type="button" class="w-full py-2 bg-background border border-error/30 text-error font-mono text-[0.65rem] font-bold uppercase tracking-wider hover:bg-error/10 transition-all" id="btn-delete-person" data-pid="' + TraceClient.escapeHtml(p.person_id) + '">Terminate Record</button>'
        + '</div>'
        + '</div>';

      // Wire delete
      var delBtn = $("btn-delete-person");
      if (delBtn) {
        delBtn.addEventListener("click", function () {
          var pid = delBtn.getAttribute("data-pid");
          TraceDialog.confirm(
            "Terminate Record",
            "Confirm record termination for " + pid + "? This action is irreversible.",
            { type: "error", confirmText: "Terminate" }
          ).then(function(ok) {
            if (!ok) return;
            TraceClient.deletePerson(pid).then(function () {
              TraceToast.success("Record Terminated", "Entity record " + pid + " has been permanently deleted.");
              loadPersonList();
              panel.innerHTML = '<div class="flex flex-col items-center justify-center h-full text-center opacity-30">'
                + '<span class="material-symbols-outlined text-[48px] mb-2">delete_forever</span>'
                + '<p class="font-mono text-[0.65rem] uppercase tracking-widest">Record terminated</p></div>';
            });
          });
        });
      }    });
  }

  function detailRow(label, value, valueClass) {
    return '<div class="flex justify-between items-baseline">'
      + '<span class="stat-label">' + label + '</span>'
      + '<span class="text-[0.7rem] ' + (valueClass || "text-white") + ' text-right">' + TraceClient.escapeHtml(value || "—") + '</span>'
      + '</div>';
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

    var clearImgBtn = $("btn-clear-images");
    if (clearImgBtn) {
      clearImgBtn.addEventListener("click", function(e) {
        e.stopPropagation();
        _selectedFiles = [];
        renderThumbnails();
        updateCreateButton();
      });
    }
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
    var previewArea = $("preview-area");
    
    if (countEl) countEl.textContent = _selectedFiles.length + " SAMPLES SELECTED";
    if (previewArea) {
      if (_selectedFiles.length > 0) previewArea.classList.remove("hidden");
      else previewArea.classList.add("hidden");
    }
    
    if (!grid) return;

    grid.innerHTML = "";
    _selectedFiles.forEach(function (file, idx) {
      var wrapper = document.createElement("div");
      wrapper.className = "relative group cursor-pointer aspect-square overflow-hidden border border-outline-variant/10";
      
      var img = document.createElement("img");
      img.alt = file.name;
      img.className = "w-full h-full object-cover grayscale hover:grayscale-0 transition-all duration-300";
      var url = URL.createObjectURL(file);
      img.src = url;
      
      var overlay = document.createElement("div");
      overlay.className = "absolute inset-0 bg-error/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity";
      overlay.innerHTML = '<span class="material-symbols-outlined text-white">delete</span>';
      
      wrapper.appendChild(img);
      wrapper.appendChild(overlay);
      
      wrapper.addEventListener("click", function (e) {
        e.stopPropagation();
        _selectedFiles.splice(idx, 1);
        renderThumbnails();
        updateCreateButton();
      });
      grid.appendChild(wrapper);
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
    if (statusEl) statusEl.textContent = "COMMITTING ENROLLMENT...";

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
        if (statusEl) statusEl.textContent = "ENROLLMENT FAILED";
        if (btn) btn.disabled = false;
        return;
      }
      if (statusEl) statusEl.textContent = "UPLOADING BIOMETRICS...";

      // Upload images
      TraceClient.uploadPersonImages(result.person_id, _selectedFiles).then(function (uploadResult) {
        if (uploadResult) {
          if (statusEl) statusEl.textContent = "✓ " + result.person_id + " — ENROLLED";
          TraceToast.success("Enrollment Successful", "Entity " + payload.name + " (" + result.person_id + ") registered with " + _selectedFiles.length + " samples.");
        } else {
          if (statusEl) statusEl.textContent = "RECORD CREATED, BIOMETRIC UPLOAD FAILED";
          TraceToast.warning("Partial Success", "Record created, but biometric samples failed to upload.");
        }
        setTimeout(function() { if(statusEl) statusEl.textContent = ""; }, 4000);
        clearForm();
        loadPersonList();
      });
    }).catch(function () {
      if (statusEl) statusEl.textContent = "SYSTEM ERROR";
      TraceToast.error("System Error", "Failed to communicate with enrollment service.");
      if (btn) btn.disabled = false;
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

  function initResizableIndex() {
    var left = $("enroll-left");
    var splitter = $("enroll-splitter");
    var toggle = $("index-toggle");
    var reopen = $("index-reopen");
    if (!left || !splitter || !toggle) return;

    // Toggle logic for LEFT panel (index)
    toggle.addEventListener("click", function() {
      left.classList.toggle("collapsed");
    });

    // Reopen button logic for LEFT panel
    if (reopen) {
      reopen.addEventListener("click", function() {
        left.classList.remove("collapsed");
      });
    }

    // ── RIGHT panel (Intelligence) toggle ────────────────────────────────
    var intelPanel  = $("enroll-right");
    var center      = $("enroll-center");
    var intelToggle = $("intel-toggle");
    var intelReopen = $("intel-reopen");

    function _setIntelCollapsed(isCollapsed) {
      if (!intelPanel) return;
      intelPanel.classList.toggle("collapsed", isCollapsed);
      // Show reopen arrow on center header when panel is hidden
      if (center) center.classList.toggle("intel-hidden", isCollapsed);
    }

    if (intelToggle) {
      intelToggle.addEventListener("click", function() {
        _setIntelCollapsed(true);
      });
    }

    if (intelReopen) {
      intelReopen.addEventListener("click", function() {
        _setIntelCollapsed(false);
      });
    }

    // Keyboard shortcut (Ctrl+I) toggles left index panel
    document.addEventListener("keydown", function(e) {
      if (e.ctrlKey && e.key.toLowerCase() === "i") {
        e.preventDefault();
        toggle.click();
      }
    });

    // Resizer logic
    var isDragging = false;
    splitter.addEventListener("mousedown", function(e) {
      isDragging = true;
      splitter.classList.add("dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    });

    document.addEventListener("mousemove", function(e) {
      if (!isDragging) return;
      var containerRect = document.querySelector(".enroll-grid").getBoundingClientRect();
      var newWidth = e.clientX - containerRect.left;
      
      // Constraints: 240px to 600px
      if (newWidth >= 240 && newWidth <= 600) {
        document.documentElement.style.setProperty("--enroll-left-width", newWidth + "px");
      }
    });

    document.addEventListener("mouseup", function() {
      if (!isDragging) return;
      isDragging = false;
      splitter.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    });
  }

  /* ─── Horizontal index splitter (forensic-style) ─── */
  function initIndexHSplitter() {
    var listEl    = $("person-list-root");
    var hSplitter = $("index-hsplitter");
    if (!listEl || !hSplitter) return;

    var COLLAPSED_H  = 0;          // fully collapsed
    var SNAP_THRESH  = 48;         // snap to collapsed if ≤ this when releasing
    var DEFAULT_H    = 320;        // initial list height in px

    function maxH() {
      // Maximum = viewport height minus the header at top of enroll-left and footer/splitter
      var leftEl = $("enroll-left");
      if (!leftEl) return 500;
      var headerEl = leftEl.querySelector("header");
      var headerH  = headerEl ? headerEl.getBoundingClientRect().height : 90;
      return Math.floor(leftEl.getBoundingClientRect().height - headerH - 6);
    }

    function applyH(h, animate) {
      if (animate) {
        listEl.classList.add("animating");
        var onEnd = function() {
          listEl.classList.remove("animating");
          listEl.removeEventListener("transitionend", onEnd);
        };
        listEl.addEventListener("transitionend", onEnd);
      } else {
        listEl.classList.remove("animating");
      }
      listEl.style.height  = h + "px";
      listEl.style.flex    = "none"; // override flex:1 once we set explicit height
      hSplitter.title = h <= COLLAPSED_H
        ? "Double-click to expand / Drag to resize"
        : "Drag to resize / Double-click to collapse";
    }

    // Start at the default height
    applyH(DEFAULT_H, false);

    /* ── Double-click: toggle ─────── */
    hSplitter.addEventListener("dblclick", function(e) {
      e.preventDefault();
      var cur = parseFloat(listEl.style.height) || DEFAULT_H;
      applyH(cur <= COLLAPSED_H ? DEFAULT_H : COLLAPSED_H, true);
    });

    /* ── Drag to resize ─────────────── */
    var _dragging   = false;
    var _dragStartY = 0;
    var _dragStartH = 0;

    hSplitter.addEventListener("mousedown", function(e) {
      if (e.button !== 0) return;
      _dragging   = true;
      _dragStartY = e.clientY;
      _dragStartH = parseFloat(listEl.style.height) || DEFAULT_H;
      hSplitter.classList.add("is-dragging");
      document.body.style.cursor     = "ns-resize";
      document.body.style.userSelect = "none";
      e.preventDefault();
      e.stopPropagation();
    });

    document.addEventListener("mousemove", function(e) {
      if (!_dragging) return;
      // drag down = positive clientY delta = grow the list
      var delta = e.clientY - _dragStartY;       // drag down = grow
      var newH  = Math.max(COLLAPSED_H, Math.min(maxH(), _dragStartH + delta));
      listEl.style.height = newH + "px";
      listEl.style.flex   = "none";
    });

    document.addEventListener("mouseup", function() {
      if (!_dragging) return;
      _dragging = false;
      hSplitter.classList.remove("is-dragging");
      document.body.style.cursor     = "";
      document.body.style.userSelect = "";

      // Snap: if barely opened, fully collapse
      var finalH = parseFloat(listEl.style.height) || DEFAULT_H;
      if (finalH > COLLAPSED_H && finalH <= SNAP_THRESH) {
        applyH(COLLAPSED_H, true);
      }
    });
  }

  function init() {
    var mainContent = document.querySelector("main");
    if (typeof TraceRender !== "undefined" && TraceRender.initOfflineUI) {
      TraceRender.initOfflineUI(mainContent);
    }

    initResizableIndex();
    initIndexHSplitter();
    initDropZone();
    initCameraModal();
    wireSearch();

    // Wire form events
    var nameInput = $("enroll-name");
    if (nameInput) nameInput.addEventListener("input", updateCreateButton);

    var createBtn = $("btn-create-upload");
    if (createBtn) createBtn.addEventListener("click", handleCreateUpload);

    var clearBtn = $("btn-clear-form");
    if (clearBtn) clearBtn.addEventListener("click", clearForm);

    // Initial load
    TraceClient.probe().then(function (info) {
      if (info) {
        loadPersonList();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
