/**
 * TRACE-AML | Toast Notification System
 * Non-blocking HUD alerts singleton
 */
(function() {
    "use strict";

    var TraceToast = {
        _container: null,
        _defaultDuration: 5000,

        /**
         * Initialize the toast container if it doesn't exist.
         */
        _ensureContainer: function() {
            if (this._container) return;
            this._container = document.createElement("div");
            this._container.id = "toast-container";
            document.body.appendChild(this._container);
        },

        /**
         * Internal show method
         * @param {string} type - info, success, warning, error
         * @param {string} title - Short heading
         * @param {string} message - Descriptive text
         * @param {number} [duration] - Ms before auto-hide
         */
        _show: function(type, title, message, duration) {
            this._ensureContainer();
            
            var toast = document.createElement("div");
            toast.className = "toast-item toast--" + type;
            
            var icon = "info";
            if (type === "success") icon = "check_circle";
            if (type === "warning") icon = "warning";
            if (type === "error")   icon = "error";

            toast.innerHTML = 
                '<span class="material-symbols-outlined toast-icon">' + icon + '</span>' +
                '<div class="toast-content">' +
                '  <div class="toast-title">' + (title || type) + '</div>' +
                '  <div class="toast-desc">' + message + '</div>' +
                '</div>' +
                '<button class="toast-close" title="Dismiss">[X]</button>';

            // Stacking: newest at top of container (which is bottom of screen)
            this._container.appendChild(toast);

            var self = this;
            var timeoutId = null;
            var startTime = Date.now();
            var remaining = duration || this._defaultDuration;

            var dismiss = function() {
                if (timeoutId) clearTimeout(timeoutId);
                toast.classList.add("toast--exiting");
                setTimeout(function() {
                    if (toast.parentNode) toast.parentNode.removeChild(toast);
                }, 300);
            };

            var startTimer = function() {
                startTime = Date.now();
                timeoutId = setTimeout(dismiss, remaining);
            };

            var pauseTimer = function() {
                clearTimeout(timeoutId);
                remaining -= Date.now() - startTime;
            };

            // Close button
            toast.querySelector(".toast-close").onclick = dismiss;

            // Hover behavior
            toast.onmouseenter = pauseTimer;
            toast.onmouseleave = startTimer;

            // Start initial timer
            startTimer();
        },

        info: function(title, msg) { this._show("info", title, msg); },
        success: function(title, msg) { this._show("success", title, msg); },
        warning: function(title, msg) { this._show("warning", title, msg); },
        error: function(title, msg) { this._show("error", title, msg); },

        /**
         * Generic replacement for window.alert()
         * Tries to guess type based on prefix or content
         */
        alert: function(msg) {
            var m = String(msg);
            if (m.indexOf("✓") >= 0 || m.toLowerCase().indexOf("complete") >= 0 || m.toLowerCase().indexOf("success") >= 0) {
                this.success("Action Complete", m.replace("✓", "").trim());
            } else if (m.toLowerCase().indexOf("fail") >= 0 || m.toLowerCase().indexOf("error") >= 0 || m.indexOf("✗") >= 0) {
                this.error("System Error", m.replace("✗", "").trim());
            } else if (m.toLowerCase().indexOf("warn") >= 0 || m.indexOf("⚠") >= 0) {
                this.warning("Attention Required", m.replace("⚠", "").trim());
            } else {
                this.info("System Notification", m);
            }
        }
    };

    // Export globally
    window.TraceToast = TraceToast;

    var TraceDialog = {
        /**
         * Generic confirmation dialog
         * @param {string} title
         * @param {string} message
         * @param {object} [options] - { type: 'warning'|'error', confirmText: 'Confirm' }
         * @returns {Promise<boolean>}
         */
        confirm: function(title, message, options) {
            options = options || {};
            var type = options.type || "warning";
            var confirmText = options.confirmText || "Proceed";
            var verifyText = options.verifyText || null;
            var icon = (type === "error") ? "warning" : "help";

            return new Promise(function(resolve) {
                var overlay = document.createElement("div");
                overlay.id = "dialog-overlay";
                
                var inputHtml = "";
                if (verifyText) {
                    inputHtml = 
                        '<div class="mt-6 mb-2">' +
                        '  <p class="font-mono text-[0.65rem] text-outline/60 uppercase mb-2">Type <span class="text-white">' + verifyText + '</span> to confirm</p>' +
                        '  <input type="text" id="dialog-verify-input" autocomplete="off" ' +
                        '    class="w-full bg-surface-lowest border border-outline-variant/20 text-white font-mono text-[0.8rem] p-3 text-center outline-none focus:border-white/40 transition-colors uppercase tracking-[0.2em]" />' +
                        '</div>';
                }

                overlay.innerHTML = 
                    '<div class="dialog-box dialog-box--' + type + '">' +
                    '  <div class="dialog-scanline"></div>' +
                    '  <div class="corner-bl"></div><div class="corner-br"></div>' +
                    '  <span class="material-symbols-outlined dialog-icon">' + icon + '</span>' +
                    '  <div class="dialog-title">' + title + '</div>' +
                    '  <div class="dialog-message">' + message + '</div>' +
                    inputHtml +
                    '  <div class="dialog-actions">' +
                    '    <button class="dialog-btn dialog-btn--cancel" id="dialog-cancel">Cancel</button>' +
                    '    <button class="dialog-btn dialog-btn--confirm" id="dialog-confirm">' + confirmText + '</button>' +
                    '  </div>' +
                    '</div>';

                document.body.appendChild(overlay);

                var confirmBtn = overlay.querySelector("#dialog-confirm");
                var verifyInput = overlay.querySelector("#dialog-verify-input");
                
                if (verifyText && confirmBtn && verifyInput) {
                    confirmBtn.disabled = true;
                    confirmBtn.style.opacity = "0.3";
                    confirmBtn.style.cursor = "not-allowed";
                    
                    verifyInput.oninput = function() {
                        var val = verifyInput.value.trim().toUpperCase();
                        var match = (val === verifyText.toUpperCase());
                        confirmBtn.disabled = !match;
                        confirmBtn.style.opacity = match ? "1" : "0.3";
                        confirmBtn.style.cursor = match ? "pointer" : "not-allowed";
                    };
                    setTimeout(function() { verifyInput.focus(); }, 100);
                }

                var cleanup = function() {
                    if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
                };

                overlay.querySelector("#dialog-cancel").onclick = function() {
                    cleanup();
                    resolve(false);
                };

                confirmBtn.onclick = function() {
                    if (confirmBtn.disabled) return;
                    cleanup();
                    resolve(true);
                };

                // Close on overlay click (if no verifyText)
                overlay.onclick = function(e) {
                    if (e.target === overlay && !verifyText) {
                        cleanup();
                        resolve(false);
                    }
                };
            });
        }
    };

    window.TraceDialog = TraceDialog;

    // Optional: hook into window.alert if we want a complete sweep, 
    // but better to explicitly call TraceToast for better control.
})();
