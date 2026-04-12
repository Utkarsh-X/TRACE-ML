/* ═══════════════════════════════════════════════════════════════════
   TRACE-AML — Side Navigation Logic
   Collapse / expand + localStorage + Ctrl/Cmd+B + drag-resize splitter.
   ═══════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  var STORAGE_KEY   = 'trace-sidenav-collapsed';
  var STORAGE_WIDTH = 'trace-sidenav-width';
  var MIN_W         = 160;
  var MAX_W         = 380;
  var DEFAULT_W     = 240;

  function initSideNav() {
    var nav      = document.getElementById('sidenav');
    var toggle   = document.getElementById('sidenav-toggle');
    var splitter = document.getElementById('sidenav-splitter');
    if (!nav) return;

    /* ── Restore persisted state ────────────────────────── */
    var wasCollapsed = localStorage.getItem(STORAGE_KEY) === 'true';
    var savedWidth   = parseInt(localStorage.getItem(STORAGE_WIDTH), 10) || DEFAULT_W;

    if (wasCollapsed) {
      nav.classList.add('collapsed');
    } else {
      nav.style.width = savedWidth + 'px';
    }

    /* ── Expand / collapse helpers ──────────────────────── */
    function collapse() {
      nav.classList.add('collapsed');
      nav.style.width = '';        /* let CSS var handle it */
      localStorage.setItem(STORAGE_KEY, 'true');
    }
    function expand() {
      nav.classList.remove('collapsed');
      var w = parseInt(localStorage.getItem(STORAGE_WIDTH), 10) || DEFAULT_W;
      nav.style.width = w + 'px';
      localStorage.setItem(STORAGE_KEY, 'false');
    }
    function toggle_() {
      if (nav.classList.contains('collapsed')) {
        expand();
      } else {
        collapse();
      }
    }

    /* ── Hamburger button ───────────────────────────────── */
    if (toggle) toggle.addEventListener('click', toggle_);

    /* ── Keyboard shortcut: Ctrl+B / Cmd+B ─────────────── */
    document.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault();
        toggle_();
      }
    });

    /* ── Drag-resize splitter ───────────────────────────── */
    if (!splitter) return;

    var _drag = false, _startX = 0, _startW = 0;

    splitter.addEventListener('mousedown', function (e) {
      if (nav.classList.contains('collapsed')) return;  /* no resize in collapsed mode */
      if (e.button !== 0) return;
      _drag   = true;
      _startX = e.clientX;
      _startW = nav.getBoundingClientRect().width;
      splitter.classList.add('is-dragging');
      document.body.style.cursor    = 'ew-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });

    document.addEventListener('mousemove', function (e) {
      if (!_drag) return;
      var newW = Math.max(MIN_W, Math.min(MAX_W, _startW + (e.clientX - _startX)));
      nav.style.width = newW + 'px';
    });

    document.addEventListener('mouseup', function () {
      if (!_drag) return;
      _drag = false;
      splitter.classList.remove('is-dragging');
      document.body.style.cursor    = '';
      document.body.style.userSelect = '';
      var finalW = nav.getBoundingClientRect().width;
      localStorage.setItem(STORAGE_WIDTH, Math.round(finalW));
    });

    /* ── Double-click splitter → collapse / restore ─────── */
    splitter.addEventListener('dblclick', function () {
      toggle_();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSideNav);
  } else {
    initSideNav();
  }
})();
