/**
 * Entities / Subjects Page Controller
 *
 * Two-view pattern:
 *   Overview → stat tiles + searchable card grid (default landing)
 *   Detail   → profile + timeline + incidents (only after user clicks a card)
 *
 * CRUD capabilities:
 *   Edit Known   → PATCH /api/v1/entities/{id}  (metadata only)
 *   Promote Unk  → PATCH /api/v1/entities/{id}  (providing name triggers promotion)
 *   Delete Any   → DELETE /api/v1/entities/{id} (known + unknown)
 *   Add Images   → POST /api/v1/persons/{id}/images (known only, triggers auto-enroll)
 *   Replace Portrait → POST /api/v1/entities/{id}/portrait
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  var _allEntities = [];
  var _currentEntityId = null;
  var _currentProfile  = null;

  /* ─────────────────────────── View Switching ────────────────────────── */

  function showOverview() {
    $("view-overview").style.display = "block";
    $("view-detail").style.display   = "none";
  }

  function showDetail() {
    $("view-overview").style.display = "none";
    $("view-detail").style.display   = "block";
  }

  /* ─────────────────────────── Overview ─────────────────────────── */

  function loadEntityList() {
    TraceClient.entities({ limit: 300 }).then(function (list) {
      _allEntities = list || [];
      renderOverviewStats(_allEntities);
      renderGrid(_allEntities);
    }).catch(function (err) {
      console.error("Failed to load entities:", err);
      var grid = $("entity-grid");
      if (grid) {
        grid.innerHTML = '<div class="col-span-full flex flex-col items-center justify-center py-20 text-outline font-mono text-[0.75rem]">'
          + '<span class="material-symbols-outlined text-[36px] mb-3 text-error">error</span>'
          + 'Failed to load entities. Please check connection and try again.</div>';
      }
      if (window.TraceToast) TraceToast.error("Registry Error", "Failed to connect to entity service.");
    });
  }

  function renderOverviewStats(list) {
    var known   = list.filter(function (e) { return String(e.type || e.entity_type || "") === "known"; }).length;
    var unknown = list.filter(function (e) { return String(e.type || e.entity_type || "") !== "known"; }).length;
    var withInc = list.reduce(function (acc, e) {
      return acc + (parseInt(e.open_incident_count, 10) || 0);
    }, 0);

    var t = $("ov-total");     if (t) t.textContent = String(list.length);
    var k = $("ov-known");     if (k) k.textContent = String(known);
    var u = $("ov-unknown");   if (u) u.textContent = String(unknown);
    var i = $("ov-incidents"); if (i) i.textContent = String(withInc);
  }

  function renderGrid(list) {
    var grid  = $("entity-grid");
    var label = $("entity-count-label");
    if (!grid) return;
    if (label) label.textContent = list.length + " entit" + (list.length !== 1 ? "ies" : "y");

    if (list.length === 0) {
      grid.innerHTML = '<div class="col-span-full flex flex-col items-center justify-center py-20 text-outline font-mono text-[0.75rem]">'
        + '<span class="material-symbols-outlined text-[36px] mb-3">person_off</span>'
        + 'No entities found</div>';
      return;
    }

    grid.innerHTML = list.map(function (ent) {
      var isKnown   = String(ent.type || ent.entity_type || "") === "known";
      var name      = TraceClient.escapeHtml(ent.name || ent.entity_id);
      var shortId   = TraceClient.escapeHtml(String(ent.entity_id || ""));
      var cat       = String(ent.category || (isKnown ? "known" : "unknown")).toLowerCase();
      var lastSeen  = ent.last_seen_at ? TraceClient.formatTime(ent.last_seen_at) : "—";
      var openInc   = parseInt(ent.open_incident_count, 10) || 0;

      var typeKey   = cat === "criminal" ? "criminal" : cat === "vip" ? "vip" : isKnown ? "known" : "unknown";
      var badgeText = isKnown ? cat.toUpperCase() : "UNKNOWN";
      var cardType  = isKnown ? "known" : "unknown";

      // Build portrait thumbnail for the avatar cell.
      var portraitTs = Math.floor(Date.now() / 30000); // 30 s cache window
      var avatarHtml = ent.entity_id
        ? '<img src="' + TraceClient.entityPortraitUrl(ent.entity_id) + '?t=' + portraitTs + '"'
          + ' alt="" loading="lazy"'
          + ' onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'"'
          + ' style="display:block" />'
          + '<span class="material-symbols-outlined" style="font-size:18px;color:#919191;display:none">'
            + (isKnown ? 'person' : 'person_off')
          + '</span>'
        : '<span class="material-symbols-outlined" style="font-size:18px;color:#919191">'
            + (isKnown ? 'person' : 'person_off')
          + '</span>';

      return '<div class="entity-card entity-card--' + cardType + '" data-entity-id="' + TraceClient.escapeHtml(ent.entity_id) + '">'
        /* top row: badge + arrow */
        + '<div class="flex items-center justify-between mb-2">'
        +   '<span class="entity-card__badge entity-card__badge--' + typeKey + '">' + badgeText + '</span>'
        +   '<span class="ec-arrow material-symbols-outlined" style="font-size:14px;color:#919191">arrow_forward</span>'
        + '</div>'
        /* avatar + name block */
        + '<div class="flex items-center gap-3">'
        +   '<div class="entity-card__avatar">' + avatarHtml + '</div>'
        +   '<div class="flex-1 min-w-0">'
        +     '<div style="font-family:Inter,sans-serif;font-weight:600;font-size:0.9rem;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.2">' + name + '</div>'
        +     '<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#666;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + shortId + '</div>'
        +   '</div>'
        + '</div>'
        /* footer */
        + '<div class="entity-card__footer">'
        +   '<span style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#666">' + TraceClient.escapeHtml(lastSeen) + '</span>'
        +   (openInc > 0
              ? '<span style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#ffb4ab">' + openInc + ' open case' + (openInc > 1 ? 's' : '') + '</span>'
              : '<span style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#474747">No open cases</span>')
        + '</div>'
        + '</div>';
    }).join("");

    // Wire click handlers
    var cards = grid.querySelectorAll(".entity-card");
    cards.forEach(function (card) {
      card.addEventListener("click", function () {
        var eid = card.getAttribute("data-entity-id");
        if (eid) loadEntityProfile(eid);
      });
    });
  }

  /* ─────────────────────────── Filtering ────────────────────────── */

  function applyFilters() {
    var search = ($("entity-search") || {}).value || "";
    var type   = ($("entity-filter") || {}).value || "";
    var q      = search.toLowerCase();

    var filtered = _allEntities.filter(function (e) {
      var matchText = (e.name || "").toLowerCase().includes(q)
        || (e.entity_id || "").toLowerCase().includes(q)
        || (e.category || "").toLowerCase().includes(q);
      var entityType = String(e.type || e.entity_type || "");
      var matchType = !type || entityType === type;
      return matchText && matchType;
    });

    renderGrid(filtered);
  }

  /* ─────────────────────────── Detail view ──────────────────────── */

  function loadEntityProfile(entityId) {
    if (!entityId) return;
    showDetail();
    _currentEntityId = entityId;

    TraceClient.entityProfile(entityId).then(function (profile) {
      if (!profile) {
        $("entity-display-name").textContent = "Failed to load";
        return;
      }
      _currentProfile = profile;
      renderHeader(profile.entity || {}, profile.linked_person || {});
      renderStats(profile);
      renderTimeline(profile.timeline || []);
      renderIncidents(profile.incidents || []);
    });
  }

  function renderHeader(entity, person) {
    var label = $("entity-profile-label");
    if (label) label.textContent = "Entity // " + entity.entity_id;

    var nameEl = $("entity-display-name");
    if (nameEl) nameEl.textContent = entity.name || entity.entity_id || "—";

    var statusEl = $("entity-status");
    if (statusEl) statusEl.textContent = String(entity.status || "active").toUpperCase();

    var typeEl = $("entity-type");
    if (typeEl) typeEl.textContent = String(entity.category || entity.entity_type || "—").toUpperCase();

    var sevEl = $("entity-severity");
    if (sevEl) {
      var open = parseInt(entity.open_incident_count, 10) || 0;
      sevEl.textContent = open > 0 ? open + " OPEN" : "NONE";
      sevEl.className   = open > 0
        ? "text-[0.875rem] text-error uppercase font-medium"
        : "text-[0.875rem] text-on-surface-variant uppercase font-medium";
    }

    var clockEl = $("entity-clock");
    if (clockEl) clockEl.textContent = TraceClient.formatDateTime(entity.last_seen_at) || "—";

    var btnEdit = $("btn-edit-entity");
    var btnDelete = $("btn-delete-entity");
    if (btnEdit) btnEdit.style.display = "flex";
    if (btnDelete) btnDelete.style.display = "flex";

    // ── Best-match portrait ──────────────────────────────────────────
    var portraitImg = $("entity-portrait");
    var portraitPlaceholder = $("entity-portrait-placeholder");
    if (portraitImg) {
      portraitImg.classList.remove("loaded");
      portraitImg.removeAttribute("src");
    }
    if (portraitPlaceholder) portraitPlaceholder.classList.remove("hidden");

    if (entity.entity_id && portraitImg) {
      var ts = Math.floor(Date.now() / 10000); // invalidates every 10 s
      portraitImg.src = TraceClient.entityPortraitUrl(entity.entity_id) + "?t=" + ts;
    }
    // ─────────────────────────────────────────────────────────────────
  }

  function renderStats(profile) {
    var stats = profile.stats || {};

    var a = $("stat-appearances");    if (a) a.textContent = String(stats.detection_count || 0);
    var i = $("stat-incident-count"); if (i) i.textContent = String(stats.incident_count || 0);
    var r = $("stat-avg-conf");       if (r) r.textContent = String(stats.recent_alert_count || 0);

    var score = (profile.linked_person && typeof profile.linked_person.enrollment_score === "number")
      ? profile.linked_person.enrollment_score : null;
    var confEl = $("entity-confidence");
    if (confEl) confEl.textContent = score !== null ? (score * 100).toFixed(1) + "%" : "—";
    var bar = $("entity-confidence-bar");
    if (bar) bar.style.width = score !== null ? (score * 100).toFixed(1) + "%" : "0%";
  }

  function renderTimeline(timeline) {
    var root = $("entity-timeline-root");
    if (!root) return;
    if (!timeline || timeline.length === 0) {
      root.innerHTML = TraceRender.emptyState("No timeline events");
      return;
    }
    var sorted = timeline.slice().reverse().slice(0, 25);
    root.innerHTML = sorted.map(function (item) {
      var kindLabel = String(item.kind || "event").toUpperCase();
      var badgeKind = item.kind === "incident" ? "filled" : (item.kind === "alert" ? "error" : "ghost");
      var badgeHtml = TraceRender.badge(badgeKind, kindLabel);
      var time      = TraceClient.formatTime(item.timestamp_utc);
      var summary   = TraceClient.escapeHtml(item.summary || item.title || "");
      return '<div class="flex items-start gap-4 p-3 hover:bg-surface-high transition-colors">'
        + '<span class="font-mono text-[0.6rem] text-outline whitespace-nowrap mt-0.5">' + TraceClient.escapeHtml(time) + '</span>'
        + badgeHtml
        + '<div><span class="text-[0.75rem] text-on-surface">' + summary + '</span></div>'
        + '</div>';
    }).join("");
  }

  function renderIncidents(incidents) {
    var btnCases = $("btn-open-cases");
    if (btnCases) {
      if (incidents && incidents.length > 0) {
        btnCases.href = "../incidents/index.html?id=" + encodeURIComponent(incidents[0].incident_id);
      } else {
        btnCases.href = "../incidents/index.html";
      }
    }

    var root = $("entity-incidents-root");
    if (!root) return;
    if (!incidents || incidents.length === 0) {
      root.innerHTML = TraceRender.emptyState("No linked cases");
      return;
    }
    root.innerHTML = incidents.map(function (inc) {
      return TraceRender.incidentCard(inc);
    }).join("");
    // Wire click-to-incident navigation: each card opens that exact incident
    root.querySelectorAll("[data-incident-id]").forEach(function (card) {
      card.addEventListener("click", function () {
        var iid = card.getAttribute("data-incident-id");
        if (iid) window.location.href = "../incidents/index.html?id=" + encodeURIComponent(iid);
      });
    });
  }

  /* ─────────────────────────── Delete flow ──────────────────────── */

  function wireDeleteFlow() {
    var btnDelete   = $("btn-delete-entity");

    if (btnDelete) {
      btnDelete.addEventListener("click", function () {
        if (!_currentEntityId) return;
        var name = ((_currentProfile && _currentProfile.entity) || {}).name || _currentEntityId;
        
        TraceDialog.confirm(
          "Terminate Entity Record",
          "You are about to permanently delete the entity " + name + " and all associated data, including biometric embeddings, portraits, and detection history.",
          { type: "error", confirmText: "Terminate Record" }
        ).then(function(ok) {
          if (!ok) return;

          TraceClient.deleteEntity(_currentEntityId).then(function (res) {
            if (res && res.status === "deleted") {
              TraceToast.success("Entity Deleted", "Record has been permanently removed.");
              showOverview();
              loadEntityList();
            } else {
              var detail = (res && res.detail) ? String(res.detail) : "Unknown error";
              TraceToast.error("Deletion Failed", detail);
            }
          }).catch(function (err) {
            TraceToast.error("Deletion Error", "Network error or server unavailable.");
          });
        });
      });
    }
  }

  /* ─────────────────────────── Edit / Modal Tab flow ─────────────── */

  function switchModalTab(tabName) {
    var identityTab = $("edit-tab-identity");
    var imagesTab   = $("edit-tab-images");
    var tabBtns     = document.querySelectorAll(".edit-tab-btn");

    if (tabName === "identity") {
      if (identityTab) identityTab.classList.remove("hidden");
      if (imagesTab)   imagesTab.classList.add("hidden");
    } else {
      if (identityTab) identityTab.classList.add("hidden");
      if (imagesTab)   imagesTab.classList.remove("hidden");
    }

    tabBtns.forEach(function (btn) {
      var isActive = btn.getAttribute("data-tab") === tabName;
      if (isActive) {
        btn.classList.add("edit-tab-btn--active", "border-primary", "text-white");
        btn.classList.remove("border-transparent", "text-outline");
      } else {
        btn.classList.remove("edit-tab-btn--active", "border-primary", "text-white");
        btn.classList.add("border-transparent", "text-outline");
      }
    });
  }

  function openEditModal(startTab) {
    if (!_currentProfile || !_currentProfile.entity) return;
    var ent    = _currentProfile.entity;
    var person = _currentProfile.linked_person || {};
    var isKnown = String(ent.type || ent.entity_type) === "known";

    // ── Populate Identity tab ──
    var nameInput = $("edit-entity-name");
    if (nameInput) nameInput.value = ent.name || "";
    var catSel = $("edit-entity-category");
    if (catSel) catSel.value = ent.category || "unknown";
    var sevInput = $("edit-entity-severity");
    if (sevInput) sevInput.value = person.severity || ent.severity || "";
    var dobInput = $("edit-entity-dob");
    if (dobInput) dobInput.value = person.dob || ent.dob || "";
    var genderSel = $("edit-entity-gender");
    if (genderSel) genderSel.value = person.gender || ent.gender || "";
    var cityInput = $("edit-entity-city");
    if (cityInput) cityInput.value = person.last_seen_city || ent.last_seen_city || "";
    var countryInput = $("edit-entity-country");
    if (countryInput) countryInput.value = person.last_seen_country || ent.last_seen_country || "";
    var notesArea = $("edit-entity-notes");
    if (notesArea) notesArea.value = person.notes || ent.notes || "";

    // ── Modal title & desc ──
    var titleEl = $("edit-modal-title");
    var descEl  = $("edit-modal-desc");
    var saveBtn = $("btn-save-edit");
    if (!isKnown) {
      if (titleEl) titleEl.textContent = "Enroll Unknown Entity";
      if (descEl) {
        descEl.classList.remove("hidden");
        descEl.innerHTML = '👤 Provide a name and category to promote this unknown entity to a tracked identity. This will enable biometric enrollment and reference image uploads.';
      }
      if (catSel && catSel.value === "unknown") catSel.value = "criminal";
      if (saveBtn) saveBtn.innerHTML = '<span class="material-symbols-outlined text-[16px]">person_add</span> Enroll as Known';
    } else {
      if (titleEl) titleEl.textContent = "Edit Entity";
      if (descEl) descEl.classList.add("hidden");
      if (saveBtn) saveBtn.innerHTML = '<span class="material-symbols-outlined text-[16px]">save</span> Save Changes';
    }

    // ── Images tab: show correct section ──
    var knownSection   = $("images-section-known");
    var unknownSection = $("images-section-unknown");
    if (isKnown) {
      if (knownSection)   knownSection.classList.remove("hidden");
      if (unknownSection) unknownSection.classList.add("hidden");
      // Show portrait preview
      var prevImg = $("edit-portrait-preview");
      var prevPlaceholder = $("edit-portrait-placeholder");
      if (prevImg && ent.entity_id) {
        var ts = Math.floor(Date.now() / 10000);
        prevImg.src = TraceClient.entityPortraitUrl(ent.entity_id) + "?t=" + ts;
        prevImg.onload  = function () { prevImg.classList.remove("hidden"); if (prevPlaceholder) prevPlaceholder.classList.add("hidden"); };
        prevImg.onerror = function () { prevImg.classList.add("hidden");    if (prevPlaceholder) prevPlaceholder.classList.remove("hidden"); };
      }
      // Show images tab for known entities
      var tabBtnImages = $("tab-btn-images");
      if (tabBtnImages) tabBtnImages.style.display = "";
    } else {
      if (knownSection)   knownSection.classList.add("hidden");
      if (unknownSection) unknownSection.classList.remove("hidden");
      var tabBtnImages2 = $("tab-btn-images");
      // Still show the tab but content will say "enroll first"
      if (tabBtnImages2) tabBtnImages2.style.display = "";
    }

    // Reset image selection state
    resetImageSelection();

    // Switch to requested tab
    switchModalTab(startTab || "identity");

    // Focus on name input if unknown entity (better UX)
    if (!isKnown) {
      setTimeout(function() {
        var nameInput = $("edit-entity-name");
        if (nameInput) nameInput.focus();
      }, 100);
    }

    $("modal-edit-entity").showModal();
  }

  function closeEditModal() {
    var modal = $("modal-edit-entity");
    if (modal) modal.close();
    resetImageSelection();
    // Clear portrait upload status
    var ps = $("portrait-upload-status");
    if (ps) { ps.classList.add("hidden"); ps.textContent = ""; }
    // Clear images upload status
    var is = $("images-upload-status");
    if (is) { is.classList.add("hidden"); is.textContent = ""; }
  }

  /* ─────────────────────────── Image Upload (Reference Images) ────── */

  var _selectedFiles = [];

  function resetImageSelection() {
    _selectedFiles = [];
    var container = $("edit-image-preview-container");
    if (container) container.classList.add("hidden");
    var strip = $("edit-image-preview-strip");
    if (strip) { 
      strip.innerHTML = ""; 
    }
    var countEl = $("edit-image-count");
    if (countEl) countEl.textContent = "No images selected";
    var uploadBtn = $("btn-upload-images");
    if (uploadBtn) uploadBtn.disabled = true;
    var statusEl = $("images-upload-status");
    if (statusEl) { statusEl.classList.add("hidden"); statusEl.textContent = ""; }
    var fileInput = $("edit-image-file");
    if (fileInput) fileInput.value = "";
  }

  function handleFilesSelected(files) {
    var validFiles = [];
    for (var i = 0; i < files.length; i++) {
      if (files[i].type.startsWith("image/")) validFiles.push(files[i]);
    }
    if (!validFiles.length) return;

    _selectedFiles = validFiles;

    var countEl = $("edit-image-count");
    if (countEl) countEl.textContent = validFiles.length + " image" + (validFiles.length !== 1 ? "s" : "") + " selected";

    var uploadBtn = $("btn-upload-images");
    if (uploadBtn) uploadBtn.disabled = false;

    // Build preview strip with enhanced thumbnails
    var strip = $("edit-image-preview-strip");
    if (!strip) return;
    strip.classList.remove("hidden");
    strip.innerHTML = "";
    validFiles.forEach(function (file, index) {
      var reader = new FileReader();
      reader.onload = function (e) {
        // Thumbnail container with hover effects
        var thumb = document.createElement("div");
        thumb.className = "relative w-16 h-16 bg-surface-high border border-outline-variant/30 overflow-hidden flex-shrink-0 rounded-sm group cursor-pointer transition-all hover:border-outline-variant/60";
        thumb.style.filter = "grayscale(100%)";
        thumb.style.transition = "filter 0.2s ease, border-color 0.2s ease";
        
        // Image element
        var img = document.createElement("img");
        img.src = e.target.result;
        img.className = "w-full h-full object-cover";
        img.alt = "Preview " + (index + 1);
        
        // Delete button overlay (hidden by default, shown on hover)
        var deleteBtn = document.createElement("button");
        deleteBtn.className = "absolute inset-0 w-full h-full flex items-center justify-center bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity rounded-sm border-0 cursor-pointer";
        deleteBtn.innerHTML = '<span class="material-symbols-outlined text-error text-[20px]">delete</span>';
        deleteBtn.type = "button";
        deleteBtn.setAttribute("data-file-index", index);
        
        // Hover effects on thumbnail
        thumb.addEventListener("mouseenter", function () {
          thumb.style.filter = "grayscale(0%)";
        });
        thumb.addEventListener("mouseleave", function () {
          thumb.style.filter = "grayscale(100%)";
        });
        
        // Delete functionality
        deleteBtn.addEventListener("click", function (evt) {
          evt.stopPropagation();
          var fileIndex = parseInt(deleteBtn.getAttribute("data-file-index"), 10);
          _selectedFiles.splice(fileIndex, 1);
          handleFilesSelected(_selectedFiles);
        });
        
        thumb.appendChild(img);
        thumb.appendChild(deleteBtn);
        strip.appendChild(thumb);
      };
      reader.readAsDataURL(file);
    });
  }

  function wireImageUpload() {
    // File input change
    var fileInput = $("edit-image-file");
    if (fileInput) {
      fileInput.addEventListener("change", function () {
        handleFilesSelected(fileInput.files);
      });
    }

    // Clear button
    var clearBtn = $("btn-clear-images-edit");
    if (clearBtn) {
      clearBtn.addEventListener("click", resetImageSelection);
    }

    // Drag-drop zone
    var dropzone = $("edit-image-dropzone");
    if (dropzone) {
      dropzone.addEventListener("dragover", function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
      dropzone.addEventListener("dragleave", function () {
        dropzone.classList.remove("dragover");
      });
      dropzone.addEventListener("drop", function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
        handleFilesSelected(e.dataTransfer.files);
      });
      dropzone.addEventListener("click", function (e) {
        // Don't trigger when clicking the "browse files" label (it has its own for)
        if (e.target.tagName !== "LABEL" && e.target.tagName !== "INPUT") {
          var fileInput2 = $("edit-image-file");
          if (fileInput2) fileInput2.click();
        }
      });
    }

    // Upload & Enroll button
    var uploadBtn = $("btn-upload-images");
    if (uploadBtn) {
      uploadBtn.addEventListener("click", function () {
        if (!_selectedFiles.length || !_currentProfile) return;

        // We need the person_id (only available for known entities)
        var person = _currentProfile.linked_person || {};
        var personId = person.person_id || null;
        if (!personId) {
          if (window.TraceToast) TraceToast.error("Enrollment Required", "Cannot upload images for an unknown entity. Switch to Identity tab and enroll them first.");
          return;
        }

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="material-symbols-outlined text-[14px]">hourglass_empty</span> Uploading…';

        var statusEl = $("images-upload-status");
        if (statusEl) {
          statusEl.classList.remove("hidden");
          statusEl.className = "mt-3 p-3 bg-surface-high border border-outline-variant/20 rounded-sm font-mono text-[0.65rem] text-outline-variant";
          statusEl.innerHTML = '<span class="material-symbols-outlined text-[14px] align-middle mr-2">hourglass_empty</span>Uploading ' + _selectedFiles.length + ' image(s)…';
        }

        TraceClient.uploadPersonImages(personId, _selectedFiles).then(function (res) {
          if (res) {
            var msg = "Uploaded " + (res.uploaded || 0) + " image(s). " + (res.enrollment || "Enrollment queued.");
            if (statusEl) {
              statusEl.className = "mt-3 p-3 bg-green-900/20 border border-green-500/30 rounded-sm font-mono text-[0.65rem] text-green-400";
              statusEl.innerHTML = '<span class="material-symbols-outlined text-[14px] align-middle mr-2">check_circle</span>' + msg;
            }
            if (window.TraceToast) TraceToast.success("Images Uploaded", msg);
            resetImageSelection();
            // Refresh the portrait after a short delay (enrollment takes a few seconds)
            setTimeout(function () {
              loadEntityProfile(_currentEntityId);
            }, 3000);
          } else {
            if (statusEl) {
              statusEl.className = "mt-3 p-3 bg-red-900/20 border border-red-500/30 rounded-sm font-mono text-[0.65rem] text-red-400";
              statusEl.innerHTML = '<span class="material-symbols-outlined text-[14px] align-middle mr-2">error</span>Upload failed. Check file format and connection.';
            }
            if (window.TraceToast) TraceToast.error("Upload Failed", "One or more images failed to upload. Please check the file format and try again.");
          }
        }).catch(function () {
          if (statusEl) {
            statusEl.className = "mt-3 p-3 bg-red-900/20 border border-red-500/30 rounded-sm font-mono text-[0.65rem] text-red-400";
            statusEl.innerHTML = '<span class="material-symbols-outlined text-[14px] align-middle mr-2">error</span>Network error during upload. Check your connection and try again.';
          }
          if (window.TraceToast) TraceToast.error("Network Error", "Connection lost during image upload. Please check your connection and try again.");
        }).finally(function () {
          uploadBtn.disabled = _selectedFiles.length === 0;
          uploadBtn.innerHTML = '<span class="material-symbols-outlined text-[14px]">upload</span>Upload &amp; Enroll';
        });
      });
    }
  }

  /* ─────────────────────────── Portrait Upload ────────────────────── */

  function wirePortraitUpload() {
    var portraitFileInput = $("edit-portrait-file");
    if (!portraitFileInput) return;

    portraitFileInput.addEventListener("change", function () {
      if (!portraitFileInput.files || !portraitFileInput.files.length) return;
      var file = portraitFileInput.files[0];
      if (!_currentEntityId) return;

      var statusEl = $("portrait-upload-status");
      if (statusEl) {
        statusEl.classList.remove("hidden");
        statusEl.className = "mt-3 p-3 bg-surface-high border border-outline-variant/20 rounded-sm font-mono text-[0.65rem] text-outline-variant";
        statusEl.innerHTML = '<span class="material-symbols-outlined text-[14px] align-middle mr-2">hourglass_empty</span>Uploading portrait…';
      }

      TraceClient.uploadPortrait(_currentEntityId, file).then(function (res) {
        if (res && res.status === "updated") {
          if (statusEl) {
            statusEl.className = "mt-3 p-3 bg-green-900/20 border border-green-500/30 rounded-sm font-mono text-[0.65rem] text-green-400";
            statusEl.innerHTML = '<span class="material-symbols-outlined text-[14px] align-middle mr-2">check_circle</span>Portrait updated successfully. Score locked at 1.0.';
          }
          if (window.TraceToast) TraceToast.success("Portrait Updated", "Primary portrait replaced successfully. Changes are immutable.");
          // Refresh the portrait thumbnail in the modal and in the detail view
          var prevImg = $("edit-portrait-preview");
          if (prevImg) {
            prevImg.src = TraceClient.entityPortraitUrl(_currentEntityId) + "?t=" + Date.now();
          }
          // Refresh the main entity portrait after modal closes
          var mainPortrait = $("entity-portrait");
          if (mainPortrait) {
            mainPortrait.src = TraceClient.entityPortraitUrl(_currentEntityId) + "?t=" + Date.now();
          }
        } else {
          if (statusEl) {
            statusEl.className = "mt-3 p-3 bg-red-900/20 border border-red-500/30 rounded-sm font-mono text-[0.65rem] text-red-400";
            statusEl.innerHTML = '<span class="material-symbols-outlined text-[14px] align-middle mr-2">error</span>Portrait upload failed. Check file format (JPEG/PNG).';
          }
          if (window.TraceToast) TraceToast.error("Portrait Upload Failed", "Could not upload portrait. Check file format (JPEG/PNG) and try again.");
        }
      }).catch(function () {
        if (statusEl) {
          statusEl.className = "mt-3 p-3 bg-red-900/20 border border-red-500/30 rounded-sm font-mono text-[0.65rem] text-red-400";
          statusEl.innerHTML = '<span class="material-symbols-outlined text-[14px] align-middle mr-2">error</span>Network error during portrait upload.';
        }
        if (window.TraceToast) TraceToast.error("Network Error", "Connection lost during portrait upload. Please check your connection and try again.");
      }).finally(function () {
        portraitFileInput.value = "";
      });
    });
  }

  /* ─────────────────────────── Save (Identity) flow ──────────────── */

  function wireSaveFlow() {
    var btnSaveEdit = $("btn-save-edit");
    if (!btnSaveEdit) return;

    btnSaveEdit.addEventListener("click", function () {
      if (!_currentEntityId) return;
      var nameVal = ($("edit-entity-name") || {}).value || "";
      nameVal = nameVal.trim();
      var isKnown = String((_currentProfile.entity || {}).type || (_currentProfile.entity || {}).entity_type) === "known";

      if (!isKnown && !nameVal) {
        if (window.TraceToast) TraceToast.warning("Name Required", "Provide a full name to enroll this unknown entity and create a tracked identity.");
        return;
      }

      var payload = {
        name:     nameVal,
        category: ($("edit-entity-category") || {}).value || "unknown",
        severity: (($("edit-entity-severity") || {}).value || "").trim(),
        dob:      (($("edit-entity-dob") || {}).value || "").trim(),
        gender:   (($("edit-entity-gender") || {}).value || "").trim(),
        city:     (($("edit-entity-city") || {}).value || "").trim(),
        country:  (($("edit-entity-country") || {}).value || "").trim(),
        notes:    (($("edit-entity-notes") || {}).value || "").trim(),
      };

      btnSaveEdit.disabled = true;
      var origHtml = btnSaveEdit.innerHTML;
      btnSaveEdit.innerHTML = '<span class="material-symbols-outlined text-[16px]">hourglass_empty</span> Saving…';

      TraceClient.updateEntity(_currentEntityId, payload).then(function (res) {
        if (res && (res.status === "updated" || res.status === "promoted")) {
          if (window.TraceToast) {
            TraceToast.success(
              res.status === "promoted" ? "Entity Enrolled" : "Entity Updated",
              res.status === "promoted" ? "Unknown entity has been promoted to a known person with biometric enrollment." : "Entity information saved successfully."
            );
          }
          closeEditModal();

          // If promoted, the entity ID changes — update URL param and reload
          var fetchId = res.new_entity_id || _currentEntityId;
          var params = new URLSearchParams(window.location.search);
          if (params.get("id")) {
            params.set("id", fetchId);
            window.history.replaceState({}, "", "?" + params.toString());
          }

          _currentEntityId = fetchId;
          loadEntityProfile(fetchId);
          loadEntityList(); // refresh background list
        } else {
          if (window.TraceToast) TraceToast.error("Save Failed", "Could not update entity. Check backend logs and try again.");
        }
      }).catch(function () {
        if (window.TraceToast) TraceToast.error("Network Error", "Connection lost while saving changes. Please check your connection and try again.");
      }).finally(function () {
        btnSaveEdit.disabled = false;
        btnSaveEdit.innerHTML = origHtml;
      });
    });
  }


  /* ─────────────────────────── Init ─────────────────────────────── */

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    var params = new URLSearchParams(window.location.search);
    var initialId = params.get("id");

    if (initialId) {
      loadEntityProfile(initialId);
    } else {
      showOverview();
    }

    // Search + filter wiring
    var searchEl = $("entity-search");
    if (searchEl) searchEl.addEventListener("input", applyFilters);
    var filterEl = $("entity-filter");
    if (filterEl) filterEl.addEventListener("change", applyFilters);

    // Back button
    var backBtn = $("btn-back");
    if (backBtn) backBtn.addEventListener("click", function () { showOverview(); });

    // Open edit modal
    var btnEdit = $("btn-edit-entity");
    if (btnEdit) btnEdit.addEventListener("click", function () { openEditModal("identity"); });

    // Tab switching within modal
    var tabBtns = document.querySelectorAll(".edit-tab-btn");
    tabBtns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        switchModalTab(btn.getAttribute("data-tab") || "identity");
      });
    });

    // Close / cancel modal
    var btnCloseEdit  = $("btn-close-edit");
    var btnCancelEdit = $("btn-cancel-edit");
    if (btnCloseEdit)  btnCloseEdit.addEventListener("click", closeEditModal);
    if (btnCancelEdit) btnCancelEdit.addEventListener("click", closeEditModal);

    // Save identity
    wireSaveFlow();

    // Delete flow (inline confirm bar)
    wireDeleteFlow();

    // Image upload on Images tab
    wireImageUpload();

    // Portrait replace on Images tab
    wirePortraitUpload();

    // Load entity list on start
    loadEntityList();
    TraceClient.probe(); // fire-and-forget for connection badge

    // Auto-refresh every 10 s when overview is visible
    setInterval(function () {
      var overview = $("view-overview");
      if (overview && overview.style.display !== "none") {
        loadEntityList();
      }
    }, 10000);

    // Re-render timestamps when timezone changes
    window.addEventListener("trace:tz-change", function () {
      if (_allEntities.length) renderGrid(_allEntities);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
