/**
 * Incidents Page Controller
 *
 * Fixes applied vs v1:
 *  - Arrow nav buttons (prev/next) scroll the card strip; sentinel still lazy-loads more
 *  - All colors now use design-system tokens (badge--neutral, badge--ghost, #ffb4ab, etc.)
 *  - Timeline cards are fully rendered with border, typed dots, proper time + meta layout
 *  - Trigger alert rows use the app's tonal / border-left pattern
 *  - Timeline items windowed at 30, "Show older" reveals more from already-fetched data
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  /* ── Constants ──────────────────────────────────── */
  var PAGE_SIZE     = 20;
  var TL_PAGE       = 30;
  var SCROLL_STEP   = 440; // px per arrow click

  /* ── State ──────────────────────────────────────── */
  var _currentId  = null;
  var _offset     = 0;
  var _done       = false;
  var _tlRaw      = []; // raw data from server
  var _tlSuppressed = []; // processed data after suppression
  var _tlShown    = 0;

  /* ════════════════════════════════════════════════
     CARD STRIP — lazy load + arrow nav
  ════════════════════════════════════════════════ */

  function loadCards(initial) {
    if (_done) return;
    TraceClient.incidents({ limit: PAGE_SIZE, skip: _offset }).then(function (list) {
      // Remove skeleton placeholders on first batch
      if (initial) {
        ["sk1","sk2","sk3"].forEach(function (id) {
          var el = $(id);
          if (el) el.parentNode.removeChild(el);
        });
      }

      if (!list || list.length === 0) {
        _done = true;
        if (initial) showEmpty();
        return;
      }

      if (list.length < PAGE_SIZE) _done = true;
      _offset += list.length;

      var strip    = $("card-strip");
      var sentinel = $("card-sentinel");
      list.forEach(function (inc) {
        strip.insertBefore(buildCard(inc), sentinel);
      });

      // Do NOT auto-select first card on initial load
      // Page should start clean/empty until user explicitly clicks a card
      // This allows the timeline placeholder "Select an incident above" to remain visible

      updateArrows();
    });
  }

  function buildCard(inc) {
    var div = document.createElement("div");
    div.className = "inc-card";
    div.setAttribute("data-id", inc.incident_id);

    var sev       = String(inc.severity || "low");
    var sevBadge  = sev === "high" ? "badge--neutral" : "badge--ghost";
    var status    = String(inc.status  || "open").toUpperCase();
    /* Show last 8 chars of incident ID so it stays compact */
    var shortId   = TraceClient.escapeHtml(String(inc.incident_id || "").slice(-8));
    /* Highlight the prefix (e.g. "REAPPEARANCE:") if present */
    var summary = TraceClient.escapeHtml(inc.summary || "Incident");
    if (summary.includes(": ")) {
      var parts = summary.split(": ");
      var prefix = parts[0];
      var rest = parts.slice(1).join(": ").trim();

      // If the rest of the string starts with the same word as the prefix, remove it to avoid "TYPE: TYPE ..."
      // This handles existing redundant data without needing a database wipe.
      var prefixUpper = prefix.toUpperCase();
      if (rest.toUpperCase().startsWith(prefixUpper)) {
        // Remove the redundant prefix from the start of the rest string
        rest = rest.substring(prefix.length).trim();
        // Clean up any leading colons, dashes or spaces that might remain
        rest = rest.replace(/^[:\s-]+/, "");
        // Capitalize the first letter of the remaining sentence
        if (rest.length > 0) {
          rest = rest.charAt(0).toUpperCase() + rest.slice(1);
        }
      }

      summary = '<span class="font-bold text-on-surface">' + prefix + ':</span> ' + rest;
    }
    var entityId  = TraceClient.escapeHtml(inc.entity_id || "—");
    var timeStr   = TraceClient.escapeHtml(TraceClient.formatTime(inc.last_seen_time || inc.start_time));

    div.innerHTML =
      '<div class="inc-card__id">' + shortId + '</div>'
    + '<div class="inc-card__sum">' + summary + '</div>'
    + '<div class="inc-card__foot">'
    +   '<span class="inc-card__ent">' + entityId + '</span>'
    +   '<span class="badge ' + sevBadge + '" style="font-size:0.52rem;padding:1px 5px">' + sev.toUpperCase() + '</span>'
    + '</div>'
    + '<div class="inc-card__time mt-1">' + timeStr + ' · ' + status + '</div>';

    div.addEventListener("click", function () {
      activateCard(div);
      loadIncident(inc.incident_id);
    });
    return div;
  }

  function activateCard(card) {
    var strip = $("card-strip");
    if (strip) {
      var prev = strip.querySelector(".inc-card.active");
      if (prev) prev.classList.remove("active");
    }
    card.classList.add("active");
  }

  function showEmpty() {
    var el = $("no-incidents"); if (el) el.classList.remove("hidden");
    var b  = $("inc-body");    if (b)  b.classList.add("hidden");
  }

  /* ── Arrow navigation ───────────────────────────── */

  function updateArrows() {
    var strip = $("card-strip");
    var prev  = $("strip-prev");
    var next  = $("strip-next");
    if (!strip) return;
    if (prev) prev.disabled = (strip.scrollLeft <= 0);
    if (next) next.disabled = _done && (strip.scrollLeft + strip.clientWidth >= strip.scrollWidth - 4);
  }

  function initArrows() {
    var strip = $("card-strip");
    var prev  = $("strip-prev");
    var next  = $("strip-next");
    if (!strip || !prev || !next) return;

    prev.addEventListener("click", function () {
      strip.scrollBy({ left: -SCROLL_STEP, behavior: "smooth" });
      setTimeout(updateArrows, 350);
    });

    next.addEventListener("click", function () {
      strip.scrollBy({ left: SCROLL_STEP, behavior: "smooth" });
      setTimeout(updateArrows, 350);
    });

    strip.addEventListener("scroll", updateArrows);
  }

  /* ── IntersectionObserver for lazy card loading ── */

  function initSentinel() {
    var sentinel = $("card-sentinel");
    var strip    = $("card-strip");
    if (!sentinel || !strip || !("IntersectionObserver" in window)) return;

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting && !_done) loadCards(false);
      });
    }, { root: strip, threshold: 0.1 });

    observer.observe(sentinel);
  }

  /* ════════════════════════════════════════════════
     INCIDENT DETAIL
  ════════════════════════════════════════════════ */

  function loadIncident(id) {
    if (!id) return;
    _currentId = id;
    // Show left panel with incident details
    var leftPanel = $("inc-left-panel");
    if (leftPanel) leftPanel.style.display = "flex";
    // Show skeleton in timeline while loading
    var tlEl = $("tl-events");
    if (tlEl) {
      tlEl.classList.remove("hidden");
      // Clear all items including the "Show more" button during skeleton phase
      tlEl.innerHTML =
        '<div class="skeleton mb-2" style="height:72px;opacity:.35"></div>'
      + '<div class="skeleton mb-2" style="height:72px;opacity:.2"></div>'
      + '<div class="skeleton" style="height:72px;opacity:.1"></div>';
    }

    var ph = $("tl-placeholder"); if (ph) ph.style.display = "none";
    var sm = $("tl-show-more");   if (sm) sm.classList.add("hidden");

    TraceClient.incident(id).then(function (detail) {
      if (!detail) {
        if (tlEl) tlEl.innerHTML = '<div class="text-outline font-mono text-[0.7rem] py-12 text-center">Incident not found or failed to load</div>';
        setText("ctrl-status", "Load failed");
        return;
      }
      renderHeader(detail.incident);
      renderEntity(detail.entity);
      renderAlerts(detail.alerts);
      
      // Store raw timeline and render with current suppression
      _tlRaw = detail.timeline ? detail.timeline.slice().reverse() : [];
      renderTimeline();

      renderActions(detail.actions);
      syncControls(detail.incident);
      handleResolution(detail.entity);
    }).catch(function(err) {
      if (tlEl) tlEl.innerHTML = '<div class="text-outline font-mono text-[0.7rem] py-12 text-center">Error loading incident data</div>';
      setText("ctrl-status", "Connection error");
    });
  }

  /* ════════════════════════════════════════════════
     IDENTITY RESOLUTION
  ════════════════════════════════════════════════ */

  function handleResolution(entity) {
    var panel = $("inc-resolution-panel");
    if (!panel) return;
    
    // Only show for unknown entities
    if (!entity || String(entity.type || "").toLowerCase() !== "unknown") {
      panel.classList.add("hidden");
      return;
    }
    
    panel.classList.remove("hidden");
    var root = $("inc-resolution-root");
    root.innerHTML = '<div class="text-outline font-mono text-[0.62rem] py-2 text-center">Searching for matches...</div>';
    
    TraceClient.entitySuggestions(entity.entity_id).then(function(list) {
      if (!list || list.length === 0) {
        root.innerHTML = '<div class="text-outline font-mono text-[0.62rem] py-2 text-center">No similar persons found</div>';
        return;
      }
      
      root.innerHTML = list.map(function(s) {
        var pct = Math.round(s.similarity * 100);
        var name = TraceClient.escapeHtml(s.name);
        var pid = TraceClient.escapeHtml(s.person_id);
        var color = pct > 70 ? "text-primary" : "text-amber-200/80";
        
        return '<div class="bg-surface-lowest/50 border border-outline-variant/10 p-2 flex items-center justify-between group hover:border-outline-variant/30 transition-all">'
          + '<div class="min-w-0">'
          +   '<div class="flex items-center gap-2">'
          +     '<span class="font-mono text-[0.65rem] font-medium text-on-surface truncate">' + name + '</span>'
          +     '<span class="font-mono text-[0.55rem] ' + color + '">' + pct + '% match</span>'
          +   '</div>'
          +   '<div class="font-mono text-[0.52rem] text-outline mt-0.5">' + pid + '</div>'
          + '</div>'
          + '<button type="button" class="btn-merge-entity px-2 py-1 bg-surface-high hover:bg-primary hover:text-on-primary border border-outline-variant/30 font-mono text-[0.55rem] uppercase rounded transition-all active:scale-95" data-target="' + s.person_id + '">'
          +   'Link'
          + '</button>'
          + '</div>';
      }).join("");
      
      root.querySelectorAll(".btn-merge-entity").forEach(function(btn) {
        btn.addEventListener("click", function() {
          var targetId = btn.getAttribute("data-target");
          TraceDialog.confirm(
            "Merge Intelligence",
            "Merge this unknown entity into " + targetId + "?\n\nAll detection history and incidents will be linked to this person.",
            { confirmText: "Merge" }
          ).then(function(ok) {
            if (!ok) return;
            btn.disabled = true;
            btn.textContent = "...";
            TraceClient.entityMerge(entity.entity_id, targetId).then(function(res) {
              if (res) {
                setText("ctrl-status", "Identity resolved");
                // Refresh list and select same index or reload
                resetStrip();
                loadCards(true);
              } else {
                btn.disabled = false;
                btn.textContent = "Link";
                if (window.TraceToast) {
                  window.TraceToast.error("Merge Failed", "Could not link incidents. Check backend logs.");
                } else {
                  console.error("Merge failed. Check backend logs.");
                }
              }
            });
          });
        });
      });    });
  }

  /* ════════════════════════════════════════════════
     RENDER HELPERS
  ════════════════════════════════════════════════ */

  function renderHeader(inc) {
    if (!inc) return;
    var shortId = String(inc.incident_id || "").slice(-8);
    setText("inc-id-label",    "Case #" + shortId);
    setText("inc-title",       inc.summary || ("Incident " + inc.incident_id));
    var statusEl = $("inc-status");
    if (statusEl) {
      statusEl.textContent = String(inc.status || "open").toUpperCase();
      statusEl.className   = inc.status === "closed"
        ? "font-mono text-[0.72rem] text-outline"
        : "font-mono text-[0.72rem] text-primary";
    }
    setText("inc-alert-count", String(inc.alert_count || 0));
    setText("inc-start",       TraceClient.formatTime(inc.start_time));
  }

  function renderEntity(e) {
    if (!e) return;
    setText("ent-name",  e.name || e.entity_id || "—");
    var typeEl = $("ent-type");
    if (typeEl) typeEl.textContent = String(e.type || e.entity_type || "unknown").toUpperCase();
    setText("ent-first", TraceClient.formatDateTime(e.created_at)   || "—");
    setText("ent-last",  TraceClient.formatDateTime(e.last_seen_at) || "—");
    setText("ent-dets",  String(e.detection_count || e.recent_alert_count || 0));
  }

  function renderAlerts(alerts) {
    var root = $("inc-alerts-root");
    if (!root) return;
    if (!alerts || alerts.length === 0) {
      root.innerHTML = '<div class="text-outline font-mono text-[0.62rem] py-3 text-center">No alerts linked</div>';
      return;
    }

    var TYPE_ICONS = {
      'REAPPEARANCE':       '👤',
      'UNKNOWN_RECURRENCE': '❓',
      'INSTABILITY':        '⚡',
    };
    var TYPE_BORDER = {
      'REAPPEARANCE':       'border-l-amber-400/60',
      'UNKNOWN_RECURRENCE': 'border-l-rose-500/60',
      'INSTABILITY':        'border-l-orange-400/60',
    };

    root.innerHTML = alerts.slice(0, 8).map(function (a) {
      var sev    = String(a.severity || 'low');
      var type   = String(a.type || 'ALERT').toUpperCase();
      var reason = TraceClient.escapeHtml(a.reason || '');
      var time   = TraceClient.escapeHtml(TraceClient.formatTime(a.timestamp_utc));
      var cnt    = a.event_count ? ' · ' + a.event_count + ' events' : '';
      var icon   = TYPE_ICONS[type] || '🔔';
      var border = TYPE_BORDER[type] || 'border-l-outline/30';
      var alertId = TraceClient.escapeHtml(String(a.alert_id || ''));
      var isAck  = a.acknowledged;

      return '<div class="alert-row alert-row--' + sev + ' border-l-2 ' + border
        + (isAck ? ' opacity-40' : '') + '" data-alert-id="' + alertId + '">'
        + '<div class="flex items-center justify-between mb-0.5">'
        +   '<span class="font-mono text-[0.6rem] font-medium text-on-surface">' + icon + ' ' + TraceClient.escapeHtml(type) + '</span>'
        +   '<div class="flex items-center gap-2">'
        +     '<span class="font-mono text-[0.55rem] text-outline">' + time + TraceClient.escapeHtml(cnt) + '</span>'
        +     (isAck
                ? '<span class="font-mono text-[0.5rem] text-outline/50">✓ ack</span>'
                : '<button class="btn-ack-alert font-mono text-[0.5rem] text-outline/60 hover:text-primary cursor-pointer border border-outline-variant/20 px-1.5 py-0.5 transition-colors" data-id="' + alertId + '">Ack</button>'
              )
        +   '</div>'
        + '</div>'
        + '<p class="text-[0.68rem] text-on-surface-variant leading-snug">' + reason + '</p>'
        + '</div>';
    }).join('');

    // Wire acknowledge buttons
    root.querySelectorAll('.btn-ack-alert').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var aid = btn.getAttribute('data-id');
        if (!aid) return;
        fetch('/api/v1/alerts/' + encodeURIComponent(aid) + '/acknowledge', { method: 'PATCH' })
          .then(function(r) { return r.json(); })
          .then(function(data) {
            if (data.status === 'acknowledged') {
              var row = root.querySelector('[data-alert-id="' + aid + '"]');
              if (row) row.classList.add('opacity-40');
              btn.replaceWith(document.createTextNode('✓ ack'));
              if (window.TraceToast) TraceToast.success('Alert Acknowledged', '');
            }
          })
          .catch(function() {
            if (window.TraceToast) TraceToast.error('Acknowledge Failed', 'Check backend.');
          });
      });
    });
  }

  /* ── Event Grouping (Suppression) ──────────────── */

  function getSuppressedTimeline(rawEvents, windowMinutes) {
    if (windowMinutes === 0) return rawEvents;
    
    var windowMs = windowMinutes * 60000;
    var result = [];
    if (rawEvents.length === 0) return result;

    var currentGroup = null;

    rawEvents.forEach(function(ev) {
      var evTime = new Date(ev.timestamp_utc).getTime();
      
      if (!currentGroup) {
        currentGroup = {
          ev: ev,
          count: 1,
          startTime: evTime,
          endTime: evTime
        };
      } else {
        var diff = Math.abs(evTime - currentGroup.startTime);
        var sameKind = (ev.kind === currentGroup.ev.kind);
        var sameTitle = (ev.title === currentGroup.ev.title);

        if (diff <= windowMs && sameKind && sameTitle) {
          currentGroup.count++;
          currentGroup.endTime = evTime;
        } else {
          result.push(currentGroup);
          currentGroup = {
            ev: ev,
            count: 1,
            startTime: evTime,
            endTime: evTime
          };
        }
      }
    });

    if (currentGroup) result.push(currentGroup);
    return result;
  }

  function renderTimeline() {
    var win = parseInt(($("tl-suppress-select") || {}).value || "1");
    _tlSuppressed = getSuppressedTimeline(_tlRaw, win);
    _tlShown = 0;
    
    var root = $("tl-events");
    if (!root) return;
    
    // Clear previous entries but keep the "Show more" button wrapper if it exists
    root.innerHTML = 
      '<div id="tl-show-more" class="hidden pt-2 pb-4">'
    +   '<button type="button" id="btn-show-more-tl" class="w-full font-mono text-[0.62rem] text-outline uppercase tracking-widest hover:text-white transition-colors bg-transparent border border-outline-variant/20 cursor-pointer py-2.5 flex items-center justify-center gap-2">'
    +     '<span class="material-symbols-outlined text-[14px]">expand_more</span>'
    +     'Show older events'
    +   '</button>'
    + '</div>';
    
    // Re-wire the new button
    var moreBtn = $("btn-show-more-tl");
    if (moreBtn) moreBtn.addEventListener("click", function () {
      appendTlBatch(TL_PAGE);
      updateTlMore();
    });

    if (_tlSuppressed.length === 0) {
      var div = document.createElement("div");
      div.className = "tl-entry text-outline font-mono text-[0.7rem] py-12 text-center";
      div.textContent = "No timeline events";
      root.insertBefore(div, $("tl-show-more"));
      updateTlMore();
      return;
    }
    appendTlBatch(TL_PAGE);
    updateTlMore();
  }

  function appendTlBatch(count) {
    var root  = $("tl-events");
    if (!root) return;
    var moreBtn = $("tl-show-more");
    var batch = _tlSuppressed.slice(_tlShown, _tlShown + count);
    batch.forEach(function (item) {
      var div = document.createElement("div");
      div.className = "tl-entry";
      div.innerHTML = buildTlCard(item);
      root.insertBefore(div, moreBtn);
    });
    _tlShown += batch.length;
    var label = $("tl-count-label");
    if (label) label.textContent = _tlShown + " / " + _tlSuppressed.length + " entries";
  }

  function buildTlCard(item) {
    // Handle both raw events and grouped objects
    var ev = item.ev || item;
    var kind = String(ev.kind || "event");
    var count = item.count || 1;
    
    /* Badge + dot use design-system classes only */
    var badgeHtml = "";
    if      (kind === "incident") badgeHtml = '<span class="badge badge--filled">INCIDENT</span>';
    else if (kind === "alert")    badgeHtml = '<span class="badge badge--neutral">ALERT</span>';
    else if (kind === "action")   badgeHtml = '<span class="badge badge--ghost">ACTION</span>';
    else                          badgeHtml = '<span class="badge badge--ghost">EVENT</span>';

    var title   = TraceClient.escapeHtml(ev.title || "");
    if (count > 1) {
      title = '<span class="text-primary mr-1">[' + count + 'x]</span>' + title;
    }

    var summary = TraceClient.escapeHtml(ev.summary || "");
    
    var timeStr;
    var dateStr;
    
    if (count > 1) {
      var earliest = Math.min(item.startTime, item.endTime);
      var latest   = Math.max(item.startTime, item.endTime);
      var start = TraceClient.formatTime(new Date(earliest).toISOString());
      var end   = TraceClient.formatTime(new Date(latest).toISOString());
      timeStr = start + " — " + end;
      dateStr = TraceClient.formatDateTime(new Date(earliest).toISOString()).slice(0, 10);
    } else {
      timeStr = TraceClient.escapeHtml(TraceClient.formatTime(ev.timestamp_utc));
      dateStr = TraceClient.escapeHtml(TraceClient.formatDateTime(ev.timestamp_utc).slice(0, 10));
    }

    /* Build meta pills - only for non-grouped to avoid confusion if they differ */
    var meta = [];
    if (count === 1) {
      if (ev.entity_id) meta.push("Entity: " + TraceClient.escapeHtml(ev.entity_id));
      if (ev.source)    meta.push("Source: " + TraceClient.escapeHtml(ev.source));
      if (ev.metadata) {
        if (ev.metadata.track_id)   meta.push("Track: "  + TraceClient.escapeHtml(ev.metadata.track_id));
        if (ev.metadata.event_count) meta.push("Events: " + ev.metadata.event_count);
      }
    }

    return '<div class="tl-dot tl-dot--' + kind + '"></div>'
      + '<div class="tl-card tl-card--' + kind + '">'
      /* Row 1: badge + title + time */
      + '<div class="flex items-center justify-between gap-2 mb-1.5">'
      +   '<div class="flex items-center gap-2 min-w-0">'
      +     badgeHtml
      +     '<span class="font-mono text-[0.7rem] font-medium text-on-surface truncate">' + title + '</span>'
      +   '</div>'
      +   '<div class="flex-shrink-0 text-right">'
      +     '<span class="font-mono text-[0.65rem] text-white block">' + timeStr + '</span>'
      +     '<span class="font-mono text-[0.52rem] text-outline block">' + dateStr + '</span>'
      +   '</div>'
      + '</div>'
      /* Row 2: summary */
      + (summary
        ? '<p class="text-[0.7rem] text-on-surface-variant leading-relaxed mb-1.5">' + summary + '</p>'
        : '')
      /* Row 3: meta chips */
      + (meta.length
        ? '<div class="flex flex-wrap gap-2 mt-1">'
          + meta.map(function(m){ return '<span class="font-mono text-[0.55rem] text-outline border border-outline-variant/20 px-1.5 py-0.5">' + m + '</span>'; }).join("")
          + '</div>'
        : '')
      + '</div>';
  }

  function updateTlMore() {
    var btn = $("tl-show-more");
    if (!btn) return;
    if (_tlShown < _tlSuppressed.length) btn.classList.remove("hidden");
    else btn.classList.add("hidden");
  }

  function renderActions(actions) {
    var root = $("inc-actions-root");
    if (!root) return;
    if (!actions || actions.length === 0) {
      root.innerHTML = '<div class="text-outline font-mono text-[0.62rem] py-3 text-center">No actions recorded</div>';
      return;
    }

    var ACTION_ICONS = {
      'LOG':        '📋',
      'EMAIL':      '📧',
      'WHATSAPP':   '💬',
      'PDF_REPORT': '📄',
      'ALARM':      '⚠️',
    };

    root.innerHTML = actions.slice(0, 10).map(function (a) {
      var rawType = String(a.action_type || a.type || 'LOG').toUpperCase();
      var type    = TraceClient.escapeHtml(rawType);
      var trig    = TraceClient.escapeHtml(a.trigger || '');
      var time    = TraceClient.escapeHtml(TraceClient.formatTime(a.timestamp_utc));
      var id      = TraceClient.escapeHtml(String(a.action_id || '').slice(-6));
      var status  = String(a.status || 'success').toLowerCase();
      var icon    = ACTION_ICONS[rawType] || '⚙️';
      var statusColor = status === 'success' ? 'text-success' : 'text-error';
      var statusDot   = status === 'success'
        ? '<span class="w-1.5 h-1.5 rounded-full bg-success inline-block mr-1"></span>'
        : '<span class="w-1.5 h-1.5 rounded-full bg-error inline-block mr-1"></span>';

      // Pull PDF path from context if present
      var ctx = a.context || {};
      var pdfPath   = typeof ctx === 'string' ? '' : (ctx.pdf_report_path || '');
      var htmlUrl   = '';
      if (pdfPath) {
        // Derive html_url from pdf path to the exported file
        htmlUrl = pdfPath.replace(/\.pdf$/, '.html');
      }
      var reportLink = '';
      if (rawType === 'PDF_REPORT' && status === 'success' && a.reason && a.reason.indexOf('pdf_generated:') === 0) {
        var fname = a.reason.replace('pdf_generated:', '');
        // Try to build the /ui/exports URL
        reportLink = '<a href="#" class="pdf-report-link font-mono text-[0.52rem] text-primary hover:underline mt-0.5 block" '
          + 'data-pdf-name="' + TraceClient.escapeHtml(fname) + '">📄 View Report</a>';
      }

      return '<div class="bg-surface-lowest border-l-2 ' + (status === 'success' ? 'border-l-success/40' : 'border-l-error/40') + ' pl-2.5 pr-2 py-2 mb-1.5">'
        + '<div class="flex items-center justify-between">'
        +   '<span class="font-mono text-[0.6rem] font-medium text-on-surface">' + icon + ' ' + type + '</span>'
        +   '<div class="flex items-center gap-1.5">'
        +     statusDot
        +     '<span class="font-mono text-[0.55rem] ' + statusColor + '">' + status.toUpperCase() + '</span>'
        +     '<span class="font-mono text-[0.5rem] text-outline ml-1">' + time + '</span>'
        +   '</div>'
        + '</div>'
        + (trig ? '<p class="font-mono text-[0.58rem] text-on-surface-variant mt-0.5">' + trig + '</p>' : '')
        + '<span class="font-mono text-[0.52rem] text-outline block mt-0.5">' + id + '</span>'
        + reportLink
        + '</div>';
    }).join('');

    // Wire report links → open companion HTML in new tab
    root.querySelectorAll('.pdf-report-link').forEach(function(link) {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        var fname = link.getAttribute('data-pdf-name');
        if (!fname) return;
        // Scan /api/v1/incidents/{id}/reports to get html_url
        var incId = _currentId || '';
        if (!incId) return;
        fetch('/api/v1/incidents/' + encodeURIComponent(incId) + '/reports')
          .then(function(r) { return r.json(); })
          .then(function(list) {
            var match = list.find(function(r) { return r.filename === fname || r.filename === fname + '.pdf'; });
            if (match && match.html_url) {
              window.open(match.html_url, '_blank');
            } else if (list && list[0] && list[0].html_url) {
              window.open(list[0].html_url, '_blank');
            } else {
              if (window.TraceToast) TraceToast.warning('Report Not Found', 'File may have been deleted.');
            }
          })
          .catch(function() { if (window.TraceToast) TraceToast.error('Error', 'Could not load report list.'); });
      });
    });
  }

  function syncControls(inc) {
    var el = $("sev-select");
    if (el) el.value = inc.severity || "low";
    setText("ctrl-status", "");
  }

  /* ── Utility ──────────────────────────────────── */
  function setText(id, val) { var el = $(id); if (el) el.textContent = val; }

  function resetStrip() {
    var strip = $("card-strip");
    if (!strip) return;
    var cards = strip.querySelectorAll(".inc-card");
    cards.forEach(function (c) { c.parentNode.removeChild(c); });
    _offset = 0; _done = false;
  }

  /* ════════════════════════════════════════════════
     FORENSIC LOG PANEL
  ════════════════════════════════════════════════ */

  var _flBytes = 0;
  var _flCollapsed = false;

  function flAppendLine(event) {
    var body = document.getElementById('inc-fl-body');
    if (!body) return;

    var lineHtml = TraceRender.terminalLine(
      event.topic || 'SESSION',
      event.payload || {},
      event.timestamp_utc
    );

    body.insertAdjacentHTML('afterbegin', lineHtml);
    _flBytes += lineHtml.length;
    var buf = document.getElementById('inc-fl-buffer');
    if (buf) buf.textContent = Math.round(_flBytes / 1024);
    var lines = body.querySelectorAll('.log-line');
    if (lines.length > 300) lines[lines.length - 1].remove();
  }

  function initForensicPanel() {
    var MINIMIZED_H = 33;       // header-only height (px)
    var MAX_H_RATIO = 0.30;     // max 30% of viewport
    var SNAP_THRESH = 48;       // snap to minimized if opened less than this many px

    var panel     = document.getElementById('inc-forensic-panel');
    var splitter  = document.getElementById('inc-forensic-splitter');
    var toggleBtn = document.getElementById('inc-fl-toggle-btn');
    var clearBtn  = document.getElementById('inc-fl-clear');
    var body      = document.getElementById('inc-fl-body');
    var bufEl     = document.getElementById('inc-fl-buffer');

    if (!panel || !splitter) return;

    /* ── Height helpers ──────────────────────────────── */
    function maxH() { return Math.floor(window.innerHeight * MAX_H_RATIO); }

    function applyHeight(h, animate) {
      if (animate) {
        panel.classList.add('animating');
        var onEnd = function() {
          panel.classList.remove('animating');
          panel.removeEventListener('transitionend', onEnd);
        };
        panel.addEventListener('transitionend', onEnd);
      } else {
        panel.classList.remove('animating');
      }
      panel.style.height = h + 'px';
      var expanded = h > MINIMIZED_H;
      panel.classList.toggle('is-expanded', expanded);
      var icon = document.getElementById('inc-fl-toggle-icon');
      var text = document.getElementById('inc-fl-toggle-text');
      if (icon) icon.textContent = expanded ? 'expand_more' : 'expand_less';
      if (text) text.textContent = expanded ? 'COLLAPSE' : 'EXPAND';
    }

    /* Start minimized */
    applyHeight(MINIMIZED_H, false);

    /* ── Toggle button ───────────────────────────────── */
    if (toggleBtn) {
      toggleBtn.addEventListener('click', function() {
        var cur = parseFloat(panel.style.height) || MINIMIZED_H;
        applyHeight(cur <= MINIMIZED_H ? maxH() : MINIMIZED_H, true);
      });
    }

    /* ── Double-click splitter ───────────────────────── */
    splitter.addEventListener('dblclick', function(e) {
      e.preventDefault();
      var cur = parseFloat(panel.style.height) || MINIMIZED_H;
      applyHeight(cur <= MINIMIZED_H ? maxH() : MINIMIZED_H, true);
    });

    /* ── Drag to resize ──────────────────────────────── */
    var _drag = false, _startY = 0, _startH = 0;

    splitter.addEventListener('mousedown', function(e) {
      if (e.button !== 0) return;
      _drag = true;
      _startY = e.clientY;
      _startH = parseFloat(panel.style.height) || MINIMIZED_H;
      splitter.classList.add('is-dragging');
      document.body.style.cursor = 'ns-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
      if (!_drag) return;
      var newH = Math.max(MINIMIZED_H, Math.min(maxH(), _startH + (_startY - e.clientY)));
      panel.style.height = newH + 'px';
      var expanded = newH > MINIMIZED_H;
      panel.classList.toggle('is-expanded', expanded);
      var icon = document.getElementById('inc-fl-toggle-icon');
      var text = document.getElementById('inc-fl-toggle-text');
      if (icon) icon.textContent = expanded ? 'expand_more' : 'expand_less';
      if (text) text.textContent = expanded ? 'COLLAPSE' : 'EXPAND';
    });

    document.addEventListener('mouseup', function() {
      if (!_drag) return;
      _drag = false;
      splitter.classList.remove('is-dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      var finalH = parseFloat(panel.style.height) || MINIMIZED_H;
      if (finalH > MINIMIZED_H && finalH < MINIMIZED_H + SNAP_THRESH) {
        applyHeight(MINIMIZED_H, true);
      }
    });

    /* ── Clear console ───────────────────────────────── */
    if (clearBtn) {
      clearBtn.addEventListener('click', function() {
        if (body) body.innerHTML = '';
        _flBytes = 0;
        if (bufEl) bufEl.textContent = '0';
      });
    }

    /* ── Connect SSE ─────────────────────────────────── */
    TraceClient.connectSSE(function(event) { 
      // Only display meaningful events; skip metrics (session.state)
      if (TraceClient.isMeaningfulEvent(event)) {
        flAppendLine(event);
      }
    });
  }

  /* ════════════════════════════════════════════════
     CONTROLS wiring
  ════════════════════════════════════════════════ */

  function wireControls() {
    var applyBtn = $("btn-apply-sev");
    if (applyBtn) applyBtn.addEventListener("click", function () {
      if (!_currentId) return;
      var sev = ($("sev-select") || {}).value || "low";
      setText("ctrl-status", "Updating…");
      TraceClient.setSeverity(_currentId, sev).then(function (result) {
        setText("ctrl-status", result ? ("Severity → " + sev) : "Failed — offline");
      });
    });

    var closeBtn = $("btn-close-inc");
    if (closeBtn) closeBtn.addEventListener("click", function () {
      if (!_currentId) return;
      setText("ctrl-status", "Closing…");
      TraceClient.closeIncident(_currentId).then(function (result) {
        if (result) {
          setText("ctrl-status", "Closed");
          resetStrip();
          loadCards(true);
        } else {
          setText("ctrl-status", "Failed — offline");
        }
      });
    });

    var genReportBtn = $("btn-gen-report");
    if (genReportBtn) genReportBtn.addEventListener("click", function () {
      if (!_currentId) return;
      genReportBtn.disabled = true;
      setText("ctrl-status", "Generating report…");
      fetch('/api/v1/incidents/' + encodeURIComponent(_currentId) + '/report', { method: 'POST' })
        .then(function(r) { return r.json(); })
        .then(function(data) {
          genReportBtn.disabled = false;
          if (data.status === 'generated') {
            setText("ctrl-status", "Report ready");
            if (window.TraceToast) {
              TraceToast.success(
                'Report Generated',
                data.html_url ? 'Opening in new tab…' : 'PDF saved to exports.'
              );
            }
            if (data.html_url) {
              setTimeout(function() { window.open(data.html_url, '_blank'); }, 400);
            }
            // Reload actions panel to show new PDF_REPORT entry
            setTimeout(function() {
              TraceClient.getActions(_currentId).then(function(acts) {
                if (acts) renderActions(acts);
              });
            }, 1000);
          } else {
            setText("ctrl-status", "Report failed: " + (data.detail || 'unknown'));
            if (window.TraceToast) TraceToast.error('Report Failed', data.detail || data.reason || '');
          }
        })
        .catch(function(e) {
          genReportBtn.disabled = false;
          setText("ctrl-status", "Network error");
          if (window.TraceToast) TraceToast.error('Report Error', String(e));
        });
    });

    // Suppression selector wiring
    var suppressSelect = $("tl-suppress-select");
    if (suppressSelect) {
      suppressSelect.addEventListener("change", function() {
        if (_tlRaw && _tlRaw.length > 0) {
          renderTimeline();
        }
      });
    }

    var moreBtn = $("btn-show-more-tl");
    if (moreBtn) moreBtn.addEventListener("click", function () {
      appendTlBatch(TL_PAGE);
      updateTlMore();
    });
  }

  /* ════════════════════════════════════════════════
     INIT
  ════════════════════════════════════════════════ */

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    wireControls();
    initArrows();
    initSentinel();
    initForensicPanel();

    // Load cards immediately — no probe() gate
    loadCards(true);
    TraceClient.probe(); // fire-and-forget for connection badge
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
