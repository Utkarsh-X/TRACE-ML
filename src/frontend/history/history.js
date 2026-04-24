/**
 * History Page Controller — Reference Design
 *
 * - Global_Density bar chart (monochromatic, reference-inspired)
 * - Date-grouped timeline cards
 * - Tactical filter bar: kind, suppression, time range, export
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  /* ─── Filters ─────────────────────────────────────────────── */

  function getFilters() {
    var kind = ($("filter-kind") || {}).value || "";
    return {
      kinds: kind ? [kind] : undefined,
      limit: 1000,
    };
  }

  /* ─── Suppression grouping ─────────────────────────────────── */

  function getSuppressedTimeline(rawEvents, windowMinutes) {
    if (windowMinutes === 0) return rawEvents;
    var windowMs = windowMinutes * 60000;
    var result = [];
    if (rawEvents.length === 0) return result;
    var currentGroup = null;
    rawEvents.forEach(function (ev) {
      var evTime = new Date(ev.timestamp_utc).getTime();
      if (!currentGroup) {
        currentGroup = { ev: ev, count: 1, startTime: evTime, endTime: evTime };
      } else {
        var diff = Math.abs(evTime - currentGroup.startTime);
        var sameKind  = ev.kind  === currentGroup.ev.kind;
        var sameTitle = ev.title === currentGroup.ev.title;
        if (diff <= windowMs && sameKind && sameTitle) {
          currentGroup.count++;
          currentGroup.endTime = evTime;
        } else {
          result.push(currentGroup);
          currentGroup = { ev: ev, count: 1, startTime: evTime, endTime: evTime };
        }
      }
    });
    if (currentGroup) result.push(currentGroup);
    return result;
  }

  /* ─── Load ─────────────────────────────────────────────────── */

  function loadTimeline() {
    var filters = getFilters();
    TraceClient.globalTimeline(filters).then(function (items) {
      renderAll(items);
    });
  }

  /* ─── Insight Strip ────────────────────────────────────────── */

  function renderInsightStrip(items) {
    var totalEl   = $("ins-total");
    var alertsEl  = $("ins-alerts");
    var incEl     = $("ins-incidents");
    var peakEl    = $("ins-peak");
    var peakSubEl = $("ins-peak-count");
    var windowEl  = $("ins-window");
    if (!totalEl) return;

    if (!items || items.length === 0) {
      totalEl.textContent  = "0";
      alertsEl.textContent = "0";
      incEl.textContent    = "0";
      peakEl.textContent   = "—";
      return;
    }

    var total = 0, alerts = 0, incidents = 0;
    var BUCKETS = 24;
    var timestamps = items.map(function (it) {
      return it.startTime || new Date((it.ev || it).timestamp_utc).getTime();
    });
    var minT  = Math.min.apply(null, timestamps);
    var maxT  = Math.max.apply(null, timestamps);
    var range = maxT - minT || 1;
    var buckets     = new Array(BUCKETS).fill(0);
    var bucketTimes = [];
    for (var bi = 0; bi < BUCKETS; bi++) {
      bucketTimes.push(minT + (bi / (BUCKETS - 1)) * range);
    }

    items.forEach(function (it) {
      var ev = it.ev || it;
      var n  = it.count || 1;
      total += n;
      var k  = (ev.kind || "").toLowerCase();
      if (k.indexOf("alert")    !== -1) alerts    += n;
      if (k.indexOf("incident") !== -1) incidents += n;
      var t  = it.startTime || new Date(ev.timestamp_utc).getTime();
      var bi = Math.min(BUCKETS - 1, Math.floor(((t - minT) / range) * (BUCKETS - 1)));
      buckets[bi] += n;
    });

    var maxC    = Math.max.apply(null, buckets);
    var peakIdx = buckets.indexOf(maxC);
    var peakD   = new Date(bucketTimes[peakIdx]);
    var peakStr = String(peakD.getHours()).padStart(2, "0") + ":" +
                  String(peakD.getMinutes()).padStart(2, "0");

    totalEl.textContent  = total;
    alertsEl.textContent = alerts;
    incEl.textContent    = incidents;
    peakEl.textContent   = peakStr;
    if (peakSubEl) peakSubEl.textContent = maxC + " events/bucket";

    if (windowEl) {
      var fmtD = function (ms) {
        var d = new Date(ms);
        return d.getDate() + " " +
          ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][d.getMonth()];
      };
      windowEl.textContent = minT === maxT ? fmtD(minT) : fmtD(minT) + " – " + fmtD(maxT);
    }
  }

  /* ─── Global Density Chart ─────────────────────────────────── */

  function renderDensityChart(items) {
    var wrap    = $("density-chart-wrap");
    var countEl = $("density-total-count");
    var startEl = $("fbar-time-start");
    var endEl   = $("fbar-time-end");
    if (!wrap) return;

    /* ── helpers ── */
    var fmtFull = function (ms) {
      var d = new Date(ms);
      return String(d.getHours()).padStart(2,"0") + ":" +
             String(d.getMinutes()).padStart(2,"0") + ":" +
             String(d.getSeconds()).padStart(2,"0");
    };
    var fmtDate = function (ms) {
      var d = new Date(ms);
      return d.getFullYear() + "." + String(d.getMonth()+1).padStart(2,"0") + "." + String(d.getDate()).padStart(2,"0");
    };
    var MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    /* adaptive axis label — auto-selects precision based on data range */
    var fmtAxis = function (ms, dataRange) {
      var d = new Date(ms);
      var hhmm = String(d.getHours()).padStart(2,"0") + ":" + String(d.getMinutes()).padStart(2,"0");
      if (dataRange >= 86400000) {
        /* multi-day: show date + time so direction is unambiguous */
        return String(d.getDate()) + " " + MON[d.getMonth()] + " " + hhmm;
      }
      if (dataRange < 120000) {
        /* sub-2-minute: add seconds for precision */
        return hhmm + ":" + String(d.getSeconds()).padStart(2,"0");
      }
      return hhmm;
    };

    /* ── empty state ── */
    if (!items || items.length === 0) {
      wrap.innerHTML = "";
      if (countEl) countEl.textContent = "0";
      return;
    }

    /* ── bucket data ── */
    var BUCKETS = 40;
    var timestamps = items.map(function(it) {
      return it.startTime || new Date((it.ev||it).timestamp_utc).getTime();
    });
    var minT  = Math.min.apply(null, timestamps);
    var maxT  = Math.max.apply(null, timestamps);
    var range = maxT - minT || 1;
    var total = 0;
    var buckets     = new Array(BUCKETS).fill(0);
    var bucketTimes = [];
    for (var bi = 0; bi < BUCKETS; bi++) {
      bucketTimes.push(minT + (bi / (BUCKETS-1)) * range);
    }
    items.forEach(function(it) {
      var t  = it.startTime || new Date((it.ev||it).timestamp_utc).getTime();
      var bi = Math.min(BUCKETS-1, Math.floor(((t-minT)/range)*(BUCKETS-1)));
      var n  = it.count || 1;
      buckets[bi] += n; total += n;
    });
    var maxC    = Math.max.apply(null, buckets) || 1;
    var peakIdx = buckets.indexOf(maxC);

    if (countEl) countEl.textContent = total.toLocaleString();
    if (startEl) startEl.textContent = fmtDate(minT);
    if (endEl)   endEl.textContent   = fmtDate(maxT);

    /* ── chart geometry ── */
    var M  = { top: 12, right: 14, bottom: 28, left: 44 };
    var CH = 210;
    var CW = wrap.offsetWidth || 700;
    var pW = CW - M.left - M.right;
    var pH = CH - M.top - M.bottom;

    var pts = buckets.map(function(c, i) {
      return {
        x:     M.left + (i / (BUCKETS-1)) * pW,
        y:     M.top  + pH - (c / maxC) * pH * 0.88,
        count: c,
        time:  bucketTimes[i]
      };
    });

    /* ── smooth bezier path helper ── */
    function smoothD(ps) {
      if (ps.length < 2) return "";
      var d = "M " + ps[0].x.toFixed(1) + " " + ps[0].y.toFixed(1);
      for (var i = 0; i < ps.length - 1; i++) {
        var mx = ((ps[i].x + ps[i+1].x) / 2).toFixed(1);
        d += " C " + mx + " " + ps[i].y.toFixed(1) +
             " "   + mx + " " + ps[i+1].y.toFixed(1) +
             " "   + ps[i+1].x.toFixed(1) + " " + ps[i+1].y.toFixed(1);
      }
      return d;
    }

    var baseY    = (M.top + pH).toFixed(1);
    var linePath = smoothD(pts);
    var areaPath = linePath +
      " L " + pts[pts.length-1].x.toFixed(1) + " " + baseY +
      " L " + pts[0].x.toFixed(1) + " " + baseY + " Z";

    /* ── build SVG ── */
    var o = [];
    o.push('<svg id="density-svg" viewBox="0 0 ' + CW + ' ' + CH + '" width="100%" height="' + CH + '" xmlns="http://www.w3.org/2000/svg" style="display:block;overflow:visible;">');

    /* gradient fill */
    o.push('<defs><linearGradient id="lc-fill-grad" x1="0" y1="0" x2="0" y2="1">');
    o.push('<stop offset="0%" stop-color="rgba(255,255,255,0.09)"/>');
    o.push('<stop offset="100%" stop-color="rgba(255,255,255,0.00)"/>');
    o.push('</linearGradient></defs>');

    /* grid lines + Y labels */
    for (var g = 0; g <= 4; g++) {
      var gy  = (M.top + (g/4) * pH).toFixed(1);
      var gv  = Math.round(maxC * (1 - g/4));
      o.push('<line x1="' + M.left + '" y1="' + gy + '" x2="' + (M.left+pW) + '" y2="' + gy + '" stroke="rgba(255,255,255,0.05)" stroke-width="1" stroke-dasharray="3,5"/>');
      o.push('<text x="' + (M.left-6) + '" y="' + gy + '" fill="rgba(255,255,255,0.2)" font-family="\'JetBrains Mono\',monospace" font-size="9" text-anchor="end" dominant-baseline="middle">' + gv + '</text>');
    }

    /* baseline */
    o.push('<line x1="' + M.left + '" y1="' + baseY + '" x2="' + (M.left+pW) + '" y2="' + baseY + '" stroke="rgba(255,255,255,0.14)" stroke-width="1"/>');

    /* bucket boundary guides — every 4 buckets, very faint */
    for (var bgi = 4; bgi < BUCKETS; bgi += 4) {
      var bgx = (M.left + (bgi / (BUCKETS-1)) * pW).toFixed(1);
      o.push('<line x1="' + bgx + '" y1="' + M.top + '" x2="' + bgx + '" y2="' + baseY + '" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>');
    }

    /* density zone labels — right edge, inside plot */
    var zones = [
      { ratio: 1.0,  label: "PEAK" },
      { ratio: 0.65, label: "HIGH" },
      { ratio: 0.33, label: "MED"  }
    ];
    zones.forEach(function(z) {
      var zy = (M.top + pH - z.ratio * pH * 0.88).toFixed(1);
      o.push('<text x="' + (M.left + pW - 3) + '" y="' + zy + '" fill="rgba(255,255,255,0.13)" font-family="\'JetBrains Mono\',monospace" font-size="7" text-anchor="end" dominant-baseline="middle">' + z.label + '</text>');
    });

    /* area fill */
    o.push('<path d="' + areaPath + '" fill="url(#lc-fill-grad)"/>');

    /* line */
    o.push('<path d="' + linePath + '" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>');

    /* markers */
    pts.forEach(function(pt, idx) {
      var isPeak = idx === peakIdx;
      if (isPeak) {
        o.push('<circle cx="' + pt.x.toFixed(1) + '" cy="' + pt.y.toFixed(1) + '" r="5.5" fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="1" class="lc-marker" data-idx="' + idx + '"/>');
        o.push('<circle cx="' + pt.x.toFixed(1) + '" cy="' + pt.y.toFixed(1) + '" r="2.5" fill="white" class="lc-marker" data-idx="' + idx + '"/>');
      } else if (pt.count > 0) {
        o.push('<circle cx="' + pt.x.toFixed(1) + '" cy="' + pt.y.toFixed(1) + '" r="2" fill="rgba(255,255,255,0.45)" stroke="rgba(255,255,255,0.15)" stroke-width="1" class="lc-marker" data-idx="' + idx + '"/>');
      } else {
        o.push('<circle cx="' + pt.x.toFixed(1) + '" cy="' + pt.y.toFixed(1) + '" r="1.2" fill="rgba(255,255,255,0.12)" class="lc-marker" data-idx="' + idx + '"/>');
      }
    });

    /* hover: crosshair + ring (hidden initially) */
    o.push('<line id="lc-crosshair" x1="0" y1="' + M.top + '" x2="0" y2="' + baseY + '" stroke="rgba(255,255,255,0.22)" stroke-width="1" stroke-dasharray="3,3" display="none"/>');
    o.push('<circle id="lc-hover-ring" cx="0" cy="0" r="6.5" fill="none" stroke="rgba(255,255,255,0.55)" stroke-width="1.5" display="none"/>');
    o.push('<circle id="lc-hover-dot"  cx="0" cy="0" r="2.8" fill="white" display="none"/>');

    /* click: selection state (persistent) */
    o.push('<circle id="lc-sel-ring" cx="0" cy="0" r="8" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.75)" stroke-width="1.5" stroke-dasharray="3,2" display="none"/>');
    o.push('<circle id="lc-sel-dot"  cx="0" cy="0" r="3.2" fill="white" display="none"/>');
    /* selected bucket vertical highlight */
    o.push('<line id="lc-sel-line" x1="0" y1="' + M.top + '" x2="0" y2="' + baseY + '" stroke="rgba(255,255,255,0.18)" stroke-width="1" display="none"/>');

    /* X-axis time labels — adaptive precision based on data range */
    var xLY   = (M.top + pH + 17).toFixed(1);
    var xMid  = (M.left + pW / 2).toFixed(1);
    o.push('<text x="' + M.left + '" y="' + xLY + '" fill="rgba(255,255,255,0.2)" font-family="\'JetBrains Mono\',monospace" font-size="8" text-anchor="start">'  + fmtAxis(minT, range) + '</text>');
    o.push('<text x="' + xMid + '" y="' + xLY + '" fill="rgba(255,255,255,0.15)" font-family="\'JetBrains Mono\',monospace" font-size="8" text-anchor="middle">' + fmtAxis((minT+maxT)/2, range) + '</text>');
    o.push('<text x="' + (M.left+pW).toFixed(1) + '" y="' + xLY + '" fill="rgba(255,255,255,0.2)" font-family="\'JetBrains Mono\',monospace" font-size="8" text-anchor="end">'   + fmtAxis(maxT, range) + '</text>');

    /* invisible hit rect for mouse events */
    o.push('<rect id="lc-hit" x="' + M.left + '" y="' + M.top + '" width="' + pW + '" height="' + pH + '" fill="transparent" style="cursor:crosshair;"/>');
    o.push('</svg>');

    /* tooltip div */
    o.push('<div id="lc-tooltip" style="display:none;"></div>');

    /* store to module scope so click handler and restore can access */
    _bucketTimes  = bucketTimes;
    _chartRange   = range;
    _chartBuckets = BUCKETS;

    wrap.innerHTML = o.join("");

  /* ── hover + click wiring ── */
    var svgEl     = wrap.querySelector("#density-svg");
    var crosshair = wrap.querySelector("#lc-crosshair");
    var ring      = wrap.querySelector("#lc-hover-ring");
    var dot       = wrap.querySelector("#lc-hover-dot");
    var selRing   = wrap.querySelector("#lc-sel-ring");
    var selDot    = wrap.querySelector("#lc-sel-dot");
    var tipEl     = wrap.querySelector("#lc-tooltip");
    var hitRect   = wrap.querySelector("#lc-hit");
    if (!hitRect || !svgEl) return;

    /* restore selection ring if a bucket was previously selected */
    if (_selectedBucketIdx >= 0 && _selectedBucketIdx < pts.length) {
      var sPt = pts[_selectedBucketIdx];
      selRing.setAttribute("cx", sPt.x.toFixed(1));
      selRing.setAttribute("cy", sPt.y.toFixed(1));
      selRing.removeAttribute("display");
      selDot.setAttribute("cx", sPt.x.toFixed(1));
      selDot.setAttribute("cy", sPt.y.toFixed(1));
      selDot.removeAttribute("display");
    }

    hitRect.addEventListener("mousemove", function(e) {
      var svgRect  = svgEl.getBoundingClientRect();
      var mouseX   = e.clientX - svgRect.left;
      var relX     = mouseX - M.left;
      var idx      = Math.max(0, Math.min(BUCKETS-1, Math.round((relX / pW) * (BUCKETS-1))));
      var pt       = pts[idx];

      /* crosshair */
      crosshair.setAttribute("x1", pt.x.toFixed(1));
      crosshair.setAttribute("x2", pt.x.toFixed(1));
      crosshair.removeAttribute("display");

      /* hover ring + dot */
      ring.setAttribute("cx", pt.x.toFixed(1));
      ring.setAttribute("cy", pt.y.toFixed(1));
      ring.removeAttribute("display");
      dot.setAttribute("cx", pt.x.toFixed(1));
      dot.setAttribute("cy", pt.y.toFixed(1));
      dot.removeAttribute("display");

      /* density classification for tooltip */
      var ratio = maxC > 0 ? pt.count / maxC : 0;
      var densityLbl = pt.count === 0   ? "no activity"       :
                       ratio >= 0.9     ? "↑ peak activity"   :
                       ratio >= 0.6     ? "high density"       :
                       ratio >= 0.3     ? "moderate activity"  :
                                          "low activity";

      /* tooltip content */
      tipEl.innerHTML =
        '<div style="color:rgba(255,255,255,0.3);font-size:0.55rem;margin-bottom:2px;">' + fmtFull(pt.time) + '</div>' +
        '<div style="color:var(--primary);font-size:0.85rem;font-weight:700;letter-spacing:-0.02em;line-height:1;">' + pt.count + '</div>' +
        '<div style="color:var(--outline);font-size:0.55rem;margin-top:2px;">' + densityLbl + '</div>' +
        '<div style="color:rgba(255,255,255,0.15);font-size:0.5rem;margin-top:4px;border-top:1px solid rgba(255,255,255,0.06);padding-top:4px;">click to filter timeline</div>';

      /* tooltip position — flip left if near right edge */
      var tX = mouseX + 14;
      var tY = Math.max(4, (pt.y / CH * wrap.offsetHeight) - 28);
      if (tX + 130 > wrap.offsetWidth) tX = mouseX - 140;
      tipEl.style.left    = tX + "px";
      tipEl.style.top     = tY + "px";
      tipEl.style.display = "block";
    });

    hitRect.addEventListener("mouseleave", function() {
      crosshair.setAttribute("display", "none");
      ring.setAttribute("display", "none");
      dot.setAttribute("display", "none");
      tipEl.style.display = "none";
    });

    hitRect.addEventListener("click", function(e) {
      var svgRect = svgEl.getBoundingClientRect();
      var mouseX  = e.clientX - svgRect.left;
      var relX    = mouseX - M.left;
      var idx     = Math.max(0, Math.min(BUCKETS-1, Math.round((relX / pW) * (BUCKETS-1))));

      /* toggle: clicking same bucket again clears filter */
      if (_selectedBucketIdx === idx) {
        clearChartFilter();
        return;
      }

      /* update module-level selection + filter */
      _selectedBucketIdx = idx;
      var bucketHalfWidth = range / (BUCKETS - 1) / 2;
      _chartFilter = {
        minT:      bucketTimes[idx] - bucketHalfWidth,
        maxT:      bucketTimes[idx] + bucketHalfWidth,
        bucketIdx: idx
      };

      /* update selection ring visuals */
      var pt = pts[idx];
      selRing.setAttribute("cx", pt.x.toFixed(1));
      selRing.setAttribute("cy", pt.y.toFixed(1));
      selRing.removeAttribute("display");
      selDot.setAttribute("cx", pt.x.toFixed(1));
      selDot.setAttribute("cy", pt.y.toFixed(1));
      selDot.removeAttribute("display");

      /* show filter indicator bar */
      var filterBar   = $("chart-filter-bar");
      var filterLabel = $("chart-filter-label");
      if (filterBar)   filterBar.style.display = "flex";
      if (filterLabel) filterLabel.textContent  = "Bucket: " + fmtFull(bucketTimes[idx]) + " (" + pts[idx].count + " events)";

      /* re-render only the timeline (not the chart) */
      renderTimeline(getWindowedItems());
    });
  }

  /* ─── Date-Grouped Timeline ────────────────────────────────── */

  function fmtDayKey(ts) {
    var d = new Date(ts);
    return d.getFullYear() + "-" +
           String(d.getMonth() + 1).padStart(2, "0") + "-" +
           String(d.getDate()).padStart(2, "0");
  }

  function fmtDayTitle(dayKey) {
    var parts = dayKey.split("-");
    var months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];
    return String(parseInt(parts[2], 10)).padStart(2, "0") +
           "_" + months[parseInt(parts[1], 10) - 1] +
           "_" + parts[0].slice(2);
  }

  function fmtTime(ts) {
    var d = new Date(ts);
    return String(d.getHours()).padStart(2, "0") + ":" +
           String(d.getMinutes()).padStart(2, "0") + ":" +
           String(d.getSeconds()).padStart(2, "0");
  }

  function kindClass(kind) {
    if (!kind) return "kind-event";
    var k = kind.toLowerCase();
    if (k.indexOf("alert")    !== -1) return "kind-alert";
    if (k.indexOf("incident") !== -1) return "kind-incident";
    if (k.indexOf("action")   !== -1) return "kind-action";
    return "kind-event";
  }

  /* ─── Selection State ──────────────────────────────────────── */

  var _activeCardEl  = null;  /* currently highlighted DOM node */
  var _flatItems     = [];    /* flat ordered list used by prev/next */
  var _activeIdx     = -1;

  function selectCard(cardEl, idx) {
    /* deselect previous */
    if (_activeCardEl) _activeCardEl.classList.remove("active");
    _activeCardEl = cardEl;
    _activeIdx    = idx;
    if (cardEl) {
      cardEl.classList.add("active");
      cardEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    updatePanelNav();
  }

  function updatePanelNav() {
    var prevBtn = $("panel-prev-btn");
    var nextBtn = $("panel-next-btn");
    var posEl   = $("panel-nav-pos");
    var total   = _flatItems.length;
    if (posEl) posEl.textContent = total > 0 ? (_activeIdx + 1) + " / " + total : "— / —";
    if (prevBtn) prevBtn.disabled = _activeIdx <= 0;
    if (nextBtn) nextBtn.disabled = _activeIdx >= total - 1;
  }

  function navigatePanel(delta) {
    var newIdx = _activeIdx + delta;
    if (newIdx < 0 || newIdx >= _flatItems.length) return;
    var allCards = document.querySelectorAll(".timeline-card-item");
    var targetCard = allCards[newIdx];
    if (targetCard && targetCard.dataset.payload) {
      selectCard(targetCard, newIdx);
      showDetailsPanel(targetCard.dataset.payload);
    }
  }

  function renderTimelineCard(item, globalIdx) {
    var ev       = item.ev || item;
    var ts       = item.startTime || new Date(ev.timestamp_utc).getTime();
    var kCls     = kindClass(ev.kind);
    var title    = TraceClient.escapeHtml(ev.title || ev.kind || "Event");
    var kindLbl  = TraceClient.escapeHtml((ev.kind || "EVENT").toUpperCase().replace(/_/g," "));
    var summary  = TraceClient.escapeHtml(ev.summary || "");
    var countLbl = item.count > 1 ? "[" + item.count + "x]" : "";
    var payload  = encodeURIComponent(JSON.stringify(item));

    return '<div class="tl-card ' + kCls + ' timeline-card-item" data-payload="' + payload + '" data-idx="' + globalIdx + '">' +
      '<div class="tl-card-time">' + fmtTime(ts) + '</div>' +
      '<div class="tl-card-badge"><span class="tl-badge-pill ' + kCls + '">' + kindLbl + '</span></div>' +
      '<div class="tl-card-title">' +
        (countLbl ? '<span style="color:var(--outline);margin-right:6px;">' + countLbl + '</span>' : '') +
        title +
        (summary ? ' <span style="color:var(--outline);font-size:0.62rem;">— ' + summary + '</span>' : '') +
      '</div>' +
      '<span class="material-symbols-outlined tl-card-chevron">chevron_right</span>' +
    '</div>';
  }

  function renderTimeline(items) {
    var root = $("timeline-results");
    if (!root) return;

    renderDensityChart(items);
    renderInsightStrip(items);

    /* reset flat list + selection */
    _flatItems    = [];
    _activeCardEl = null;
    _activeIdx    = -1;
    updatePanelNav();

    /* apply chart bucket filter to timeline items only */
    var timelineItems = items;
    if (_chartFilter) {
      timelineItems = items.filter(function(it) {
        var t = it.startTime || new Date((it.ev||it).timestamp_utc).getTime();
        return t >= _chartFilter.minT && t <= _chartFilter.maxT;
      });
    }

    if (!timelineItems || timelineItems.length === 0) {
      root.innerHTML = '<div style="padding:40px 24px;font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:var(--outline);text-transform:uppercase;letter-spacing:0.12em;">' +
        (_chartFilter ? 'No events in selected bucket — click × Clear to reset' : 'No timeline events found') +
        '</div>';
      return;
    }

    var win = parseInt(($('filter-suppress') || {}).value || "1", 10);
    var suppressEl = $("filter-suppress");
    if (suppressEl) win = parseInt(suppressEl.value || "1", 10);

    timelineItems.sort(function (a, b) {
      return new Date(a.timestamp_utc).getTime() - new Date(b.timestamp_utc).getTime();
    });

    var suppressed = getSuppressedTimeline(timelineItems, win);

    /* Group by day, newest first */
    var dayMap = {}, dayOrder = [];
    suppressed.forEach(function (item) {
      var ev  = item.ev || item;
      var ts  = item.startTime || new Date(ev.timestamp_utc).getTime();
      var key = fmtDayKey(ts);
      if (!dayMap[key]) { dayMap[key] = []; dayOrder.push(key); }
      dayMap[key].push(item);
    });
    dayOrder.sort(function (a, b) { return b.localeCompare(a); });

    var todayKey = fmtDayKey(Date.now());
    var globalIdx = 0; /* continuous index across all cards for prev/next */
    _flatItems = [];

    var html = dayOrder.map(function (key) {
      var dayItems = dayMap[key].slice().reverse();
      var isOlder  = key !== todayKey;
      var titleCls = isOlder ? "tl-day-title is-older" : "tl-day-title";

      /* count breakdown for day header */
      var dayAlerts = dayItems.filter(function(it) {
        return kindClass((it.ev||it).kind) === "kind-alert";
      }).length;
      var dayIncs = dayItems.filter(function(it) {
        return kindClass((it.ev||it).kind) === "kind-incident";
      }).length;
      var daySub = [];
      if (dayIncs) daySub.push(dayIncs + " incident" + (dayIncs > 1 ? "s" : ""));
      if (dayAlerts) daySub.push(dayAlerts + " alert" + (dayAlerts > 1 ? "s" : ""));
      var dayCountStr = dayItems.length + " event" + (dayItems.length !== 1 ? "s" : "") +
                        (daySub.length ? " — " + daySub.join(", ") : "");

      var cardsHtml = dayItems.map(function(item) {
        _flatItems.push(item);
        return renderTimelineCard(item, globalIdx++);
      }).join("");

      return '<div class="tl-day-group">' +
        '<div class="tl-day-header">' +
          '<h3 class="' + titleCls + '">' + fmtDayTitle(key) + '</h3>' +
          '<div class="tl-day-rule"></div>' +
          '<span class="tl-day-count">' + dayCountStr + '</span>' +
        '</div>' +
        '<div class="tl-day-cards">' + cardsHtml + '</div>' +
      '</div>';
    }).join("");

    root.innerHTML = html;
  }

  /* ─── Detail Panel ─────────────────────────────────────────── */

  function showDetailsPanel(payloadStr) {
    try {
      var item    = JSON.parse(decodeURIComponent(payloadStr));
      var ev      = item.ev || item;
      var grid    = document.querySelector(".history-grid");
      var panel   = $("detail-panel");
      var content = $("detail-panel-content");
      if (!grid || !panel || !content) return;

      panel.style.display = "flex";
      setTimeout(function () { grid.classList.add("panel-open"); }, 10);

      var title   = TraceClient.escapeHtml(ev.title || ev.kind || "Event");
      var time    = TraceClient.formatDateTime(ev.timestamp_utc);
      var summary = TraceClient.escapeHtml(ev.summary || "");
      var reason  = ev.reason
        ? '<div style="margin-top:8px;font-family:\'Roboto Mono\',monospace;font-size:0.7rem;color:#9ca3af;">REASON: ' + TraceClient.escapeHtml(ev.reason) + "</div>"
        : "";

      var metaHtml = "";
      if (ev.metadata) {
        metaHtml = '<div style="margin-top:16px;">' +
          '<div class="stat-label" style="margin-bottom:6px;">Metadata_Payload</div>' +
          '<pre class="terminal-block" style="border:1px solid rgba(255,255,255,0.06);">' +
          TraceClient.escapeHtml(JSON.stringify(ev.metadata, null, 2)) +
          "</pre></div>";
      }

      var actionsHtml = "";
      if (ev.entity_id) {
        actionsHtml += '<a href="../entities/index.html?id=' + encodeURIComponent(ev.entity_id) + '" class="fbar-btn fbar-btn-primary" style="display:block;text-decoration:none;text-align:center;margin-bottom:4px;">View Entity</a>';
      }
      if (ev.incident_id) {
        actionsHtml += '<a href="../incidents/index.html" class="fbar-btn fbar-btn-ghost" style="display:block;text-decoration:none;text-align:center;">View Incident</a>';
      }
      var actionsContainer = actionsHtml
        ? '<div style="margin-top:20px;display:flex;flex-direction:column;gap:4px;">' + actionsHtml + "</div>"
        : "";

      var kindLbl = TraceClient.escapeHtml((ev.kind || "EVENT").toUpperCase());
      content.innerHTML =
        '<div>' +
          '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;color:var(--outline);margin-bottom:4px;text-transform:uppercase;letter-spacing:0.1em;">' + TraceClient.escapeHtml(time) + '</div>' +
          '<div style="font-family:\'Inter\',sans-serif;font-size:1rem;font-weight:700;color:var(--on-surface);line-height:1.25;margin-bottom:8px;">' + title + '</div>' +
          '<span class="badge badge--ghost" style="margin-bottom:12px;">' + kindLbl + '</span>' +
          (summary ? '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:var(--outline);line-height:1.6;">' + summary + '</p>' : '') +
          reason +
        '</div>' +
        metaHtml +
        actionsContainer;
    } catch (e) {
      console.error("Failed to parse event payload", e);
    }
  }

  function hideDetailsPanel() {
    var grid  = document.querySelector(".history-grid");
    var panel = $("detail-panel");
    if (grid) grid.classList.remove("panel-open");
    if (panel) setTimeout(function () { panel.style.display = "none"; }, 300);
  }

  /* ─── Export Log ───────────────────────────────────────────── */

  var _lastItems         = [];
  var _chartFilter       = null;   /* {minT, maxT, bucketIdx} or null */
  var _selectedBucketIdx = -1;    /* which bucket is selected in chart */
  var _bucketTimes       = [];    /* bucket center timestamps for chart */
  var _chartRange        = 1;     /* full data time range for bucket width */
  var _chartBuckets      = 40;    /* BUCKETS constant — shared with chart */
  var _timeWindow        = 'all'; /* '1h' | '1d' | '10d' | '1m' | 'all' */

  var _TW_MS = { '1h': 3600000, '1d': 86400000, '10d': 864000000, '1m': 2592000000 };

  function getWindowedItems() {
    if (_timeWindow === 'all' || !_lastItems.length) return _lastItems;
    var ms = _TW_MS[_timeWindow];
    if (!ms) return _lastItems;
    var cutoff = Date.now() - ms;
    return _lastItems.filter(function(it) {
      var t = it.startTime || new Date((it.ev || it).timestamp_utc).getTime();
      return t >= cutoff;
    });
  }

  function exportLog() {
    if (!_lastItems || _lastItems.length === 0) {
      alert("No events loaded to export.");
      return;
    }
    var rows = ["timestamp_utc,kind,title,summary,entity_id,count"];
    _lastItems.forEach(function (item) {
      var ev = item.ev || item;
      var esc = function (v) {
        var s = (v || "").toString().replace(/"/g, '""');
        return '"' + s + '"';
      };
      rows.push([
        esc(ev.timestamp_utc),
        esc(ev.kind),
        esc(ev.title || ev.kind),
        esc(ev.summary),
        esc(ev.entity_id),
        esc(item.count || 1)
      ].join(","));
    });
    var blob = new Blob([rows.join("\n")], { type: "text/csv" });
    var url  = URL.createObjectURL(blob);
    var a    = document.createElement("a");
    a.href   = url;
    a.download = "trace_aml_event_log_" + new Date().toISOString().slice(0, 10) + ".csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /* ─── renderAll ────────────────────────────────────────────── */

  function renderAll(items) {
    _lastItems = items || [];
    /* clear chart bucket filter on fresh data load */
    _chartFilter       = null;
    _selectedBucketIdx = -1;
    var filterBar = $("chart-filter-bar");
    if (filterBar) filterBar.style.display = "none";
    renderTimeline(getWindowedItems());
  }

  function clearChartFilter() {
    _chartFilter       = null;
    _selectedBucketIdx = -1;
    var filterBar = $("chart-filter-bar");
    if (filterBar) filterBar.style.display = "none";
    /* single renderTimeline call — it handles chart + strip + timeline internally */
    renderTimeline(getWindowedItems());
  }

  /* ─── Init ─────────────────────────────────────────────────── */

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    /* Primary action */
    var goBtn = $("filter-go");
    if (goBtn) goBtn.addEventListener("click", loadTimeline);

    /* Export (secondary) */
    var exportBtn = $("export-log-btn");
    if (exportBtn) exportBtn.addEventListener("click", exportLog);

    /* Filters auto-apply */
    var suppressSelect = $("filter-suppress");
    if (suppressSelect) suppressSelect.addEventListener("change", loadTimeline);

    var kindSelect = $("filter-kind");
    if (kindSelect) kindSelect.addEventListener("change", loadTimeline);

    /* Timeline card click — selection state + panel */
    var tlResults = $("timeline-results");
    if (tlResults) {
      tlResults.addEventListener("click", function (e) {
        var card = e.target.closest(".timeline-card-item");
        if (card && card.dataset.payload) {
          var idx = parseInt(card.dataset.idx || "0", 10);
          selectCard(card, idx);
          showDetailsPanel(card.dataset.payload);
        }
      });
    }

    /* Panel close */
    var closePanelBtn = $("close-panel-btn");
    if (closePanelBtn) closePanelBtn.addEventListener("click", hideDetailsPanel);

    /* Prev / Next navigation */
    var prevBtn = $("panel-prev-btn");
    if (prevBtn) prevBtn.addEventListener("click", function() { navigatePanel(-1); });
    var nextBtn = $("panel-next-btn");
    if (nextBtn) nextBtn.addEventListener("click", function() { navigatePanel(1); });

    /* Chart bucket clear filter */
    var chartClearBtn = $("chart-clear-btn");
    if (chartClearBtn) chartClearBtn.addEventListener("click", clearChartFilter);

    /* Time window pill buttons */
    var pillGroup = $("tw-pill-group");
    if (pillGroup) {
      pillGroup.addEventListener("click", function(e) {
        var pill = e.target.closest(".tw-pill");
        if (!pill) return;
        var win = pill.dataset.window;
        if (!win || win === _timeWindow) return;
        /* update active state */
        pillGroup.querySelectorAll(".tw-pill").forEach(function(p) {
          p.classList.toggle("active", p.dataset.window === win);
        });
        _timeWindow = win;
        /* changing time window clears the chart bucket filter */
        _chartFilter       = null;
        _selectedBucketIdx = -1;
        var filterBar = $("chart-filter-bar");
        if (filterBar) filterBar.style.display = "none";
        /* single renderTimeline call drives chart + strip + timeline */
        renderTimeline(getWindowedItems());
      });
    }

    TraceClient.probe().then(function (info) {
      if (info) loadTimeline();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
