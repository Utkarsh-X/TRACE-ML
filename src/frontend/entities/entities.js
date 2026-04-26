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
  var _currentProfile = null;
  var _activeQuickFilter = "all"; // tracks current quick-filter chip

  /* ─────────────────────────── View Switching ────────────────────────── */

  function showOverview() {
    $("view-overview").style.display = "block";
    $("view-detail").style.display = "none";
  }

  function showDetail() {
    $("view-overview").style.display = "none";
    $("view-detail").style.display = "block";
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
    var known = list.filter(function (e) { return String(e.type || e.entity_type || "") === "known"; }).length;
    var unknown = list.filter(function (e) { return String(e.type || e.entity_type || "") !== "known"; }).length;
    var withInc = list.reduce(function (acc, e) {
      return acc + (parseInt(e.open_incident_count, 10) || 0);
    }, 0);

    var t = $("ov-total"); if (t) t.textContent = String(list.length);
    var k = $("ov-known"); if (k) k.textContent = String(known);
    var u = $("ov-unknown"); if (u) u.textContent = String(unknown);
    var i = $("ov-incidents"); if (i) i.textContent = String(withInc);
  }

  function renderGrid(list) {
    var grid = $("entity-grid");
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
      var isKnown = String(ent.type || ent.entity_type || "") === "known";
      var name = TraceClient.escapeHtml(ent.name || ent.entity_id);
      var shortId = TraceClient.escapeHtml(String(ent.entity_id || ""));
      var cat = String(ent.category || (isKnown ? "known" : "unknown")).toLowerCase();
      var lastSeen = ent.last_seen_at ? TraceClient.formatTime(ent.last_seen_at) : "—";
      var openInc = parseInt(ent.open_incident_count, 10) || 0;

      var typeKey = cat === "criminal" ? "criminal" : cat === "vip" ? "vip" : isKnown ? "known" : "unknown";
      var badgeText = isKnown ? cat.toUpperCase() : "UNKNOWN";
      var cardType = isKnown ? "known" : "unknown";

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
        /* top row: badge + activity pill + arrow */
        + '<div class="flex items-center justify-between mb-2">'
        + '<div class="flex items-center gap-2">'
        + '<span class="entity-card__badge entity-card__badge--' + typeKey + '">' + badgeText + '</span>'
        + '</div>'
        + '<span class="ec-arrow material-symbols-outlined" style="font-size:14px;color:#919191">arrow_forward</span>'
        + '</div>'
        /* avatar + name block */
        + '<div class="flex items-center gap-3">'
        + '<div class="entity-card__avatar">' + avatarHtml + '</div>'
        + '<div class="flex-1 min-w-0">'
        + '<div style="font-family:Inter,sans-serif;font-weight:600;font-size:0.9rem;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.2">' + name + '</div>'
        + '<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#666;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + shortId + '</div>'
        + '</div>'
        + '</div>'
        /* footer */
        + '<div class="entity-card__footer">'
        + '<span style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#666">' + TraceClient.escapeHtml(lastSeen) + '</span>'
        + (openInc > 0
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

  /* ─────────────────────────── Activity helper ───────────────────── */

  function getActivityState(entity) {
    if (!entity.last_seen_at) return "idle";
    var ms = Date.now() - new Date(entity.last_seen_at).getTime();
    var openInc = parseInt(entity.open_incident_count, 10) || 0;
    if (openInc > 0) return "flagged";
    if (ms < 5 * 60 * 1000) return "active";   // <5 min
    if (ms < 24 * 60 * 60 * 1000) return "active"; // <24 h
    return "idle";
  }

  function activityPillHtml(entity) {
    var state = getActivityState(entity);
    var labels = { active: "ACTIVE", idle: "IDLE", flagged: "FLAGGED" };
    return '<span class="activity-pill activity-pill--' + state + '">' + labels[state] + '</span>';
  }

  /* ─────────────────────────── Filtering ────────────────────────── */

  function applyFilters() {
    var search = ($("entity-search") || {}).value || "";
    var q = search.toLowerCase();

    var filtered = _allEntities.filter(function (e) {
      var matchText = (e.name || "").toLowerCase().includes(q)
        || (e.entity_id || "").toLowerCase().includes(q)
        || (e.category || "").toLowerCase().includes(q);

      var entityType = String(e.type || e.entity_type || "");
      var isKnown = entityType === "known";
      var state = getActivityState(e);
      var openInc = parseInt(e.open_incident_count, 10) || 0;

      var matchQF = true;
      switch (_activeQuickFilter) {
        case "active": matchQF = state === "active"; break;
        case "incidents": matchQF = openInc > 0; break;
        case "unknown": matchQF = !isKnown; break;
        case "known": matchQF = isKnown; break;
        default: matchQF = true;
      }

      return matchText && matchQF;
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
      renderActivityGraph(profile.timeline || []);
      renderTimeline(profile.timeline || []);
      renderIncidents(profile.incidents || []);
    });
  }

  function renderHeader(entity, person) {
    var label = $("entity-profile-label");
    if (label) label.textContent = "Entity // " + entity.entity_id;

    var nameEl = $("entity-display-name");
    if (nameEl) nameEl.textContent = entity.name || entity.entity_id || "—";

    // Legacy status span (hidden, for compat)
    var statusEl = $("entity-status");
    if (statusEl) statusEl.textContent = String(entity.status || "active").toUpperCase();

    // ── FIX 4: Dominant status badge ──────────────────────────────────────
    var statusBadge = $("entity-status-badge");
    if (statusBadge) {
      var open = parseInt(entity.open_incident_count, 10) || 0;
      var rawStatus = String(entity.status || "active").toLowerCase();
      var state = open > 0 ? "flagged" : getActivityState(entity);
      var badgeConfigs = {
        flagged: { text: "⚠ FLAGGED", bg: "rgba(255,107,107,0.1)", border: "rgba(255,107,107,0.5)", color: "#ff6b6b" },
        active: { text: "● ACTIVE", bg: "rgba(74,222,128,0.07)", border: "rgba(74,222,128,0.4)", color: "#4ade80" },
        idle: { text: "◌ INACTIVE", bg: "transparent", border: "rgba(145,145,145,0.3)", color: "#919191" },
      };
      var cfg = badgeConfigs[state] || badgeConfigs.idle;
      statusBadge.textContent = cfg.text;
      statusBadge.style.background = cfg.bg;
      statusBadge.style.borderColor = cfg.border;
      statusBadge.style.color = cfg.color;
    }

    var typeEl = $("entity-type");
    if (typeEl) typeEl.textContent = String(entity.category || entity.entity_type || "—").toUpperCase();

    var sevEl = $("entity-severity");
    if (sevEl) {
      var open = parseInt(entity.open_incident_count, 10) || 0;
      sevEl.textContent = open > 0 ? open + " OPEN" : "NONE";
      sevEl.className = open > 0
        ? "text-[0.8rem] text-error uppercase font-semibold"
        : "text-[0.8rem] text-outline uppercase font-semibold";
    }

    // ── Watchlist Priority (enrolled severity) ───────────────────────────────
    var watchlistCol = $("entity-watchlist-col");
    var watchlistEl = $("entity-watchlist");
    var enrolledSev = String(person.severity || "").toLowerCase().trim();
    var sevMap = {
      "low": { label: "L1 · ROUTINE", color: "#62c6ef" },
      "medium": { label: "L2 · ELEVATED", color: "#f5a623" },
      "high": { label: "L3 · CRITICAL", color: "#ff453a" },
    };
    if (watchlistCol && watchlistEl && enrolledSev && sevMap[enrolledSev]) {
      watchlistCol.style.display = "";
      watchlistEl.textContent = sevMap[enrolledSev].label;
      watchlistEl.style.color = sevMap[enrolledSev].color;
    } else if (watchlistCol) {
      watchlistCol.style.display = "none";
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
    var entity = profile.entity || {};

    var a = $("stat-appearances"); if (a) a.textContent = String(stats.detection_count || 0);
    var i = $("stat-incident-count"); if (i) i.textContent = String(stats.incident_count || 0);
    var r = $("stat-avg-conf"); if (r) r.textContent = String(stats.recent_alert_count || 0);

    // FIX 6: Narrative last-activity text
    var actEl = $("stat-last-activity");
    if (actEl) {
      var lastSeen = entity.last_seen_at;
      actEl.textContent = lastSeen
        ? "Last activity: " + TraceClient.formatTime(lastSeen)
        : "Last activity: unknown";
    }

    // Show "—" when score is null OR effectively zero (not yet meaningful)
    var score = (profile.linked_person && typeof profile.linked_person.enrollment_score === "number")
      ? profile.linked_person.enrollment_score : null;
    var scoreValid = score !== null && score > 0.001;
    var confEl = $("entity-confidence");
    if (confEl) confEl.textContent = scoreValid ? (score * 100).toFixed(1) + "%" : "—";
    var bar = $("entity-confidence-bar");
    if (bar) bar.style.width = scoreValid ? (score * 100).toFixed(1) + "%" : "0%";
  }

  // ── Entity Activity Graph ─────────────────────────────────────────────
  // Decision-support visualization: behavior over time.
  // Computes pattern label, draws minimal area+line SVG, marks alert moments.

  function _eagSmooth(ps) {
    if (ps.length < 2) return "";
    var d = "M " + ps[0].x.toFixed(1) + " " + ps[0].y.toFixed(1);
    for (var i = 0; i < ps.length - 1; i++) {
      var mx = ((ps[i].x + ps[i + 1].x) / 2).toFixed(1);
      d += " C " + mx + " " + ps[i].y.toFixed(1) +
        " " + mx + " " + ps[i + 1].y.toFixed(1) +
        " " + ps[i + 1].x.toFixed(1) + " " + ps[i + 1].y.toFixed(1);
    }
    return d;
  }

  function _eagPattern(buckets, maxC) {
    if (!maxC) return { key: "low", label: "Low Activity" };
    var avg = buckets.reduce(function (s, v) { return s + v; }, 0) / buckets.length;
    // Variance for spread
    var variance = buckets.reduce(function (s, v) {
      return s + (v - avg) * (v - avg);
    }, 0) / buckets.length;
    var cv = avg > 0 ? Math.sqrt(variance) / avg : 0; // coefficient of variation
    if (maxC / Math.max(avg, 0.001) > 3 || cv > 1.2) {
      return { key: "burst", label: "Burst Activity" };
    }
    if (avg > 0.3) {
      return { key: "continuous", label: "Continuous Presence" };
    }
    return { key: "low", label: "Low Activity" };
  }

  function renderActivityGraph(timeline) {
    var panel = $("entity-activity-panel");
    var wrap = $("entity-activity-graph");
    var patEl = $("eag-pattern-label");
    var rangeEl = $("eag-range-label");
    if (!panel || !wrap) return;

    // Hide panel if no data
    if (!timeline || timeline.length === 0) {
      panel.style.display = "none";
      return;
    }
    panel.style.display = "";

    var BUCKETS = 30;
    var ML = 10, MR = 16, MT = 10, MB = 24;
    var CH = 130;
    var CW = wrap.offsetWidth || 800;
    var pW = CW - ML - MR;
    var pH = CH - MT - MB;

    // Compute time range from all items
    var timestamps = timeline.map(function (it) {
      return new Date(it.timestamp_utc).getTime();
    });
    var minT = Math.min.apply(null, timestamps);
    var maxT = Math.max.apply(null, timestamps);
    var range = maxT - minT || 1;

    // Bucket event counts
    var buckets = new Array(BUCKETS).fill(0);
    var bucketTimes = [];
    for (var bi = 0; bi < BUCKETS; bi++) {
      bucketTimes.push(minT + (bi / (BUCKETS - 1)) * range);
    }
    timeline.forEach(function (it) {
      var t = new Date(it.timestamp_utc).getTime();
      var b = Math.min(BUCKETS - 1, Math.floor(((t - minT) / range) * (BUCKETS - 1)));
      buckets[b] += 1;
    });

    var maxC = Math.max.apply(null, buckets) || 1;

    // Points
    var pts = buckets.map(function (c, i) {
      return {
        x: ML + (i / (BUCKETS - 1)) * pW,
        y: MT + pH - (c / maxC) * pH * 0.85,
        count: c,
        time: bucketTimes[i]
      };
    });

    var baseY = (MT + pH).toFixed(1);
    var linePath = _eagSmooth(pts);
    var areaPath = linePath +
      " L " + pts[pts.length - 1].x.toFixed(1) + " " + baseY +
      " L " + pts[0].x.toFixed(1) + " " + baseY + " Z";

    // Identify alert/incident timestamps
    var alertTimes = [];
    timeline.forEach(function (it) {
      var k = String(it.kind || "").toLowerCase();
      if (k === "alert" || k === "incident") {
        alertTimes.push(new Date(it.timestamp_utc).getTime());
      }
    });

    // Time axis formatter
    function fmtT(ms) {
      var d = new Date(ms);
      return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
    }
    function fmtRange(ms) {
      var d = new Date(ms);
      return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][d.getMonth()]
        + " " + d.getDate();
    }

    // Pattern detection
    var pattern = _eagPattern(buckets, maxC);
    if (patEl) {
      patEl.textContent = pattern.label;
      patEl.className = "eag-header__pattern eag-header__pattern--" + pattern.key;
    }
    if (rangeEl) {
      var sameDay = fmtRange(minT) === fmtRange(maxT);
      rangeEl.textContent = sameDay
        ? fmtRange(minT) + " · " + fmtT(minT) + " – " + fmtT(maxT)
        : fmtRange(minT) + " – " + fmtRange(maxT);
    }

    // Build SVG
    var o = [];
    o.push('<svg id="eag-svg" viewBox="0 0 ' + CW + ' ' + CH + '" width="100%" height="' + CH + '" xmlns="http://www.w3.org/2000/svg" style="display:block;overflow:visible;">');

    // Gradient fill
    o.push('<defs><linearGradient id="eag-fill" x1="0" y1="0" x2="0" y2="1">');
    o.push('<stop offset="0%" stop-color="rgba(255,255,255,0.07)"/>');
    o.push('<stop offset="100%" stop-color="rgba(255,255,255,0.00)"/>');
    o.push('</linearGradient></defs>');

    // Minimal grid: 2 horizontal dashed lines
    [0.33, 0.66].forEach(function (ratio) {
      var gy = (MT + ratio * pH).toFixed(1);
      o.push('<line x1="' + ML + '" y1="' + gy + '" x2="' + (ML + pW) + '" y2="' + gy +
        '" stroke="rgba(255,255,255,0.04)" stroke-width="1" stroke-dasharray="3,5"/>');
    });

    // Baseline
    o.push('<line x1="' + ML + '" y1="' + baseY + '" x2="' + (ML + pW) + '" y2="' + baseY +
      '" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>');

    // Area fill
    o.push('<path d="' + areaPath + '" fill="url(#eag-fill)"/>');

    // Line
    o.push('<path d="' + linePath + '" fill="none" stroke="rgba(255,255,255,0.55)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>');

    // Alert dot markers ON the curve
    alertTimes.forEach(function (at) {
      var b = Math.min(BUCKETS - 1, Math.floor(((at - minT) / range) * (BUCKETS - 1)));
      var pt = pts[b];
      if (!pt) return;
      // Outer ring + filled dot
      o.push('<circle cx="' + pt.x.toFixed(1) + '" cy="' + pt.y.toFixed(1) +
        '" r="4.5" fill="none" stroke="rgba(255,107,107,0.35)" stroke-width="1"/>');
      o.push('<circle cx="' + pt.x.toFixed(1) + '" cy="' + pt.y.toFixed(1) +
        '" r="2" fill="rgba(255,107,107,0.7)"/>');
    });

    // Crosshair + hover indicators (hidden initially)
    o.push('<line id="eag-xhair" x1="0" y1="' + MT + '" x2="0" y2="' + baseY +
      '" stroke="rgba(255,255,255,0.15)" stroke-width="1" stroke-dasharray="3,3" display="none"/>');
    o.push('<circle id="eag-hover-dot" cx="0" cy="0" r="3" fill="rgba(255,255,255,0.8)" display="none"/>');

    // X-axis time labels (3 pts: start, mid, end)
    var xLY = (MT + pH + 17).toFixed(1);
    o.push('<text x="' + ML + '" y="' + xLY + '" fill="rgba(255,255,255,0.18)" font-family="\'JetBrains Mono\',monospace" font-size="8" text-anchor="start">' + fmtT(minT) + '</text>');
    o.push('<text x="' + (ML + pW / 2).toFixed(1) + '" y="' + xLY + '" fill="rgba(255,255,255,0.12)" font-family="\'JetBrains Mono\',monospace" font-size="8" text-anchor="middle">' + fmtT((minT + maxT) / 2) + '</text>');
    o.push('<text x="' + (ML + pW).toFixed(1) + '" y="' + xLY + '" fill="rgba(255,255,255,0.18)" font-family="\'JetBrains Mono\',monospace" font-size="8" text-anchor="end">' + fmtT(maxT) + '</text>');

    // Invisible hit rect for mouse events
    o.push('<rect id="eag-hit" x="' + ML + '" y="' + MT + '" width="' + pW + '" height="' + pH + '" fill="transparent" style="cursor:crosshair;"/>');
    o.push('</svg>');

    // Tooltip element
    o.push('<div id="eag-tooltip" style="display:none;"></div>');

    wrap.innerHTML = o.join("");

    // ── Hover wiring ────────────────────────────────────────────────────
    var svgEl = wrap.querySelector("#eag-svg");
    var xhair = wrap.querySelector("#eag-xhair");
    var hDot = wrap.querySelector("#eag-hover-dot");
    var tipEl = wrap.querySelector("#eag-tooltip");
    var hitEl = wrap.querySelector("#eag-hit");
    if (!hitEl || !svgEl) return;

    hitEl.addEventListener("mousemove", function (e) {
      var svgRect = svgEl.getBoundingClientRect();
      var mouseX = e.clientX - svgRect.left;
      var relX = mouseX - ML;
      var idx = Math.max(0, Math.min(BUCKETS - 1, Math.round((relX / pW) * (BUCKETS - 1))));
      var pt = pts[idx];

      xhair.setAttribute("x1", pt.x.toFixed(1));
      xhair.setAttribute("x2", pt.x.toFixed(1));
      xhair.removeAttribute("display");

      hDot.setAttribute("cx", pt.x.toFixed(1));
      hDot.setAttribute("cy", pt.y.toFixed(1));
      hDot.removeAttribute("display");

      // Tooltip
      var d = new Date(pt.time);
      var timeStr = String(d.getHours()).padStart(2, "0") + ":" +
        String(d.getMinutes()).padStart(2, "0") + ":" +
        String(d.getSeconds()).padStart(2, "0");
      tipEl.innerHTML =
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.52rem;color:#555;margin-bottom:3px;">' + timeStr + '</div>' +
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.9rem;font-weight:700;color:#fff;line-height:1;">' + pt.count + '</div>' +
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.5rem;color:#444;margin-top:2px;text-transform:uppercase;">events</div>';

      var tX = (mouseX / CW * wrap.offsetWidth) + 12;
      var tY = Math.max(4, (pt.y / CH * wrap.offsetHeight) - 40);
      if (tX + 90 > wrap.offsetWidth) tX -= 100;
      tipEl.style.left = tX + "px";
      tipEl.style.top = tY + "px";
      tipEl.style.display = "block";
    });

    hitEl.addEventListener("mouseleave", function () {
      xhair.setAttribute("display", "none");
      hDot.setAttribute("display", "none");
      tipEl.style.display = "none";
    });

    // ── Click: scroll timeline to that time bucket ──────────────────────
    hitEl.addEventListener("click", function (e) {
      var svgRect = svgEl.getBoundingClientRect();
      var relX = (e.clientX - svgRect.left) - ML;
      var idx = Math.max(0, Math.min(BUCKETS - 1, Math.round((relX / pW) * (BUCKETS - 1))));
      var targetTime = bucketTimes[idx];

      // Find the first timeline row whose timestamp is nearest to targetTime
      var timelineRoot = $("entity-timeline-root");
      if (!timelineRoot) return;
      var rows = timelineRoot.querySelectorAll(".tl-header-row, .tl-event-row");
      var best = null, bestDiff = Infinity;
      rows.forEach(function (row) {
        var ts = parseInt(row.getAttribute("data-ts") || "0", 10);
        if (!ts) return;
        var diff = Math.abs(ts - targetTime);
        if (diff < bestDiff) { bestDiff = diff; best = row; }
      });
      if (best) {
        best.scrollIntoView({ behavior: "smooth", block: "center" });
        // Brief flash highlight
        best.style.background = "rgba(255,255,255,0.07)";
        setTimeout(function () { best.style.background = ""; }, 800);
      }
    });
  }

  // Timeline kind config ──────────────────────────────────────────────────────
  // "header" kinds render as prominent section-header rows.
  // Everything else renders as a compact indented event sub-row.
  var _TL_HEADER_KINDS = { "alert": true, "incident": true };

  function _buildHeaderRow(item) {
    var kind = String(item.kind || "alert").toLowerCase();
    var time = TraceClient.escapeHtml(TraceClient.formatTime(item.timestamp_utc));
    var summary = TraceClient.escapeHtml(item.summary || item.title || "");
    var modCls = kind === "incident" ? "incident" : "alert";
    var lblCls = kind === "incident" ? "incident" : "alert";
    var label = kind === "incident" ? "INCIDENT" : "ALERT";
    var ts = new Date(item.timestamp_utc).getTime();
    return '<div class="tl-header-row tl-header-row--' + modCls + '" data-ts="' + ts + '">'
      + '<span class="tl-header-row__time">' + time + '</span>'
      + '<span class="tl-header-row__label tl-header-row__label--' + lblCls + '">' + label + '</span>'
      + '<span class="tl-header-row__title">' + summary + '</span>'
      + '</div>';
  }

  function _buildEventRow(item) {
    var time = TraceClient.escapeHtml(TraceClient.formatTime(item.timestamp_utc));
    var summary = TraceClient.escapeHtml(item.summary || item.title || "");
    var ts = new Date(item.timestamp_utc).getTime();
    return '<div class="tl-event-row" data-ts="' + ts + '">'
      + '<span class="tl-event-row__time">' + time + '</span>'
      + '<span class="tl-event-row__dot"></span>'
      + '<span class="tl-event-row__text">' + summary + '</span>'
      + '</div>';
  }

  var _TIMELINE_INITIAL = 20; // rows in first render (header + events count equally)

  function renderTimeline(timeline) {
    var root = $("entity-timeline-root");
    if (!root) return;
    if (!timeline || timeline.length === 0) {
      root.innerHTML = TraceRender.emptyState("No timeline events");
      return;
    }

    var sorted = timeline.slice().reverse(); // newest first

    // Split into visible + hidden
    var visible = sorted.slice(0, _TIMELINE_INITIAL);
    var hidden = sorted.slice(_TIMELINE_INITIAL);
    var total = sorted.length;

    function buildRows(items) {
      return items.map(function (item) {
        var kind = String(item.kind || "event").toLowerCase();
        return _TL_HEADER_KINDS[kind] ? _buildHeaderRow(item) : _buildEventRow(item);
      }).join("");
    }

    var visibleHtml = buildRows(visible);

    // Count label
    var countLabel = total + " event" + (total !== 1 ? "s" : "");

    var overflowHtml = "";
    var loadMoreHtml = "";
    if (hidden.length > 0) {
      overflowHtml = '<div id="tl-overflow" class="hidden">' + buildRows(hidden) + '</div>';
      loadMoreHtml = '<button id="tl-load-more" class="tl-load-more">'
        + '<span class="material-symbols-outlined" style="font-size:13px">expand_more</span>'
        + hidden.length + ' more events'
        + '</button>';
    }

    root.innerHTML =
      // Panel header bar
      '<div class="tl-panel">'
      + '<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 14px;border-bottom:1px solid rgba(255,255,255,0.05);flex-shrink:0;">'
      + '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.55rem;text-transform:uppercase;letter-spacing:0.12em;color:#555;">' + countLabel + '</span>'
      + '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.52rem;text-transform:uppercase;letter-spacing:0.1em;color:#333;">Newest first</span>'
      + '</div>'
      // Scrollable rows area
      + '<div class="tl-scroll">'
      + visibleHtml
      + overflowHtml
      + '</div>'
      // Load more (outside scroll so it doesn't scroll away)
      + loadMoreHtml
      + '</div>';

    var loadBtn = $("tl-load-more");
    if (loadBtn) {
      loadBtn.addEventListener("click", function () {
        var overflow = $("tl-overflow");
        if (overflow) overflow.classList.remove("hidden");
        loadBtn.style.display = "none";
      });
    }
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
    var btnDelete = $("btn-delete-entity");

    if (btnDelete) {
      btnDelete.addEventListener("click", function () {
        if (!_currentEntityId) return;
        var name = ((_currentProfile && _currentProfile.entity) || {}).name || _currentEntityId;

        TraceDialog.confirm(
          "Terminate Entity Record",
          "You are about to permanently delete the entity " + name + " and all associated data, including biometric embeddings, portraits, and detection history.",
          { type: "error", confirmText: "Terminate Record" }
        ).then(function (ok) {
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
    var imagesTab = $("edit-tab-images");
    var tabBtns = document.querySelectorAll(".edit-tab-btn");

    if (tabName === "identity") {
      if (identityTab) identityTab.classList.remove("hidden");
      if (imagesTab) imagesTab.classList.add("hidden");
    } else {
      if (identityTab) identityTab.classList.add("hidden");
      if (imagesTab) imagesTab.classList.remove("hidden");
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
    var ent = _currentProfile.entity;
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
    var descEl = $("edit-modal-desc");
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
    var knownSection = $("images-section-known");
    var unknownSection = $("images-section-unknown");
    if (isKnown) {
      if (knownSection) knownSection.classList.remove("hidden");
      if (unknownSection) unknownSection.classList.add("hidden");
      // Show portrait preview
      var prevImg = $("edit-portrait-preview");
      var prevPlaceholder = $("edit-portrait-placeholder");
      if (prevImg && ent.entity_id) {
        var ts = Math.floor(Date.now() / 10000);
        prevImg.src = TraceClient.entityPortraitUrl(ent.entity_id) + "?t=" + ts;
        prevImg.onload = function () { prevImg.classList.remove("hidden"); if (prevPlaceholder) prevPlaceholder.classList.add("hidden"); };
        prevImg.onerror = function () { prevImg.classList.add("hidden"); if (prevPlaceholder) prevPlaceholder.classList.remove("hidden"); };
      }
      // Show images tab for known entities
      var tabBtnImages = $("tab-btn-images");
      if (tabBtnImages) tabBtnImages.style.display = "";
    } else {
      if (knownSection) knownSection.classList.add("hidden");
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
      setTimeout(function () {
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
        name: nameVal,
        category: ($("edit-entity-category") || {}).value || "unknown",
        severity: (($("edit-entity-severity") || {}).value || "").trim(),
        dob: (($("edit-entity-dob") || {}).value || "").trim(),
        gender: (($("edit-entity-gender") || {}).value || "").trim(),
        city: (($("edit-entity-city") || {}).value || "").trim(),
        country: (($("edit-entity-country") || {}).value || "").trim(),
        notes: (($("edit-entity-notes") || {}).value || "").trim(),
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

    // Search wiring
    var searchEl = $("entity-search");
    if (searchEl) searchEl.addEventListener("input", applyFilters);

    // Quick filter chips
    document.querySelectorAll(".quick-filter-chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        _activeQuickFilter = chip.getAttribute("data-qf") || "all";
        document.querySelectorAll(".quick-filter-chip").forEach(function (c) {
          c.classList.toggle("active", c === chip);
        });
        applyFilters();
      });
    });

    // Stat bar click-to-filter (FIX 3)
    var statMappings = [
      { id: "stat-btn-total", qf: "all" },
      { id: "stat-btn-known", qf: "known" },
      { id: "stat-btn-unknown", qf: "unknown" },
      { id: "stat-btn-incidents", qf: "incidents" },
    ];
    statMappings.forEach(function (m) {
      var btn = $(m.id);
      if (btn) btn.addEventListener("click", function () {
        _activeQuickFilter = m.qf;
        document.querySelectorAll(".quick-filter-chip").forEach(function (c) {
          c.classList.toggle("active", c.getAttribute("data-qf") === m.qf);
        });
        applyFilters();
      });
    });

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
    var btnCloseEdit = $("btn-close-edit");
    var btnCancelEdit = $("btn-cancel-edit");
    if (btnCloseEdit) btnCloseEdit.addEventListener("click", closeEditModal);
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
