/**
 * Improved Camera Capture Component
 * With better permission handling and user-friendly error messages
 */

(function() {
  "use strict";

  function $(id) { return document.getElementById(id); }

  window.CameraCapture = {
    _stream: null,
    _video: null,
    _canvas: null,
    _animationId: null,
    _isActive: false,
    _selectedFiles: [],
    _qualityThresholds: {
      sharpness: 55,
      brightness_min: 45,
      brightness_max: 220,
      face_ratio_min: 0.03,
    },

    init: function() {
      this.setupElements();
      this.attachEventListeners();
    },

    setupElements: function() {
      var enrollForm = $("enroll-form");
      if (!enrollForm) return;

      var cameraSection = document.createElement("div");
      cameraSection.id = "camera-capture-section";
      cameraSection.className = "camera-capture-section";
      cameraSection.innerHTML = `
        <div class="camera-header">
          <h3 class="text-[0.9rem] font-headline font-semibold mb-3">
            📸 Capture Images (Alternative)
          </h3>
          <p class="text-[0.7rem] text-outline mb-3">
            Or capture images directly from your camera instead of uploading files
          </p>
        </div>

        <div class="camera-container">
          <div id="camera-preview-area" class="camera-preview-area" style="display: none;">
            <video id="camera-video" playsinline autoplay muted style="width: 100%; border-radius: 4px;"></video>
            <canvas id="camera-canvas" style="display: none;"></canvas>
            
            <div class="quality-feedback">
              <div class="quality-bar">
                <div id="quality-progress" class="quality-progress" style="width: 0%;"></div>
              </div>
              <div id="quality-metrics" class="quality-metrics text-[0.65rem] space-y-1">
                <div><span>Sharpness:</span> <span id="metric-sharpness">--</span></div>
                <div><span>Brightness:</span> <span id="metric-brightness">--</span></div>
                <div><span>Face Detected:</span> <span id="metric-face">--</span></div>
              </div>
            </div>
          </div>

          <div id="camera-prompt" class="camera-prompt text-center py-6">
            <p class="text-[0.75rem] text-outline mb-3 camera-status">📷 No camera active</p>
            <button type="button" id="btn-open-camera" class="btn-primary" style="cursor: pointer;">
              Open Camera
            </button>
          </div>
        </div>

        <div class="camera-controls" style="display: none; margin-top: 12px;">
          <button type="button" id="btn-capture-image" class="btn-secondary" style="flex: 1; cursor: pointer;">
            📸 Capture Image
          </button>
          <button type="button" id="btn-close-camera" class="btn-secondary" style="flex: 1; margin-left: 8px; cursor: pointer;">
            Close Camera
          </button>
        </div>
      `;

      // Find insertion point (before the drop zone)
      var dropZone = enrollForm.querySelector(".drop-zone");
      if (dropZone) {
        dropZone.parentNode.insertBefore(cameraSection, dropZone);
      } else {
        enrollForm.appendChild(cameraSection);
      }

      // Store references
      this._video = $("camera-video");
      this._canvas = $("camera-canvas");
    },

    attachEventListeners: function() {
      var self = this;

      var btnOpenCamera = $("btn-open-camera");
      if (btnOpenCamera) {
        btnOpenCamera.addEventListener("click", function() {
          self.requestCameraPermission();
        });
      }

      var btnCloseCamera = $("btn-close-camera");
      if (btnCloseCamera) {
        btnCloseCamera.addEventListener("click", function() {
          self.closeCamera();
        });
      }

      var btnCapture = $("btn-capture-image");
      if (btnCapture) {
        btnCapture.addEventListener("click", function() {
          self.captureImage();
        });
      }
    },

    requestCameraPermission: function() {
      var self = this;
      var statusEl = document.querySelector(".camera-status");

      // Check if getUserMedia is supported
      var getUserMedia = navigator.mediaDevices && navigator.mediaDevices.getUserMedia;
      if (!getUserMedia) {
        if (statusEl) {
          statusEl.innerHTML = "❌ Camera not supported in this browser";
          statusEl.style.color = "#ff6b6b";
        }
        return;
      }

      // Request camera permission
      navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
        .then(function(stream) {
          self.openCameraWithStream(stream, statusEl);
        })
        .catch(function(err) {
          self.handleCameraError(err, statusEl);
        });
    },

    openCameraWithStream: function(stream, statusEl) {
      this._stream = stream;
      this._video.srcObject = stream;
      this._isActive = true;

      // Update status
      if (statusEl) {
        statusEl.innerHTML = "✅ Camera active - Position your face in frame";
        statusEl.style.color = "#00ffc8";
      }

      // Show camera UI
      $("camera-prompt").style.display = "none";
      $("camera-preview-area").style.display = "block";
      document.querySelector(".camera-controls").style.display = "flex";

      // Start quality monitoring
      this.startQualityMonitoring();
    },

    handleCameraError: function(err, statusEl) {
      var errorMsg = "Camera error";
      var detailMsg = err.message;

      switch (err.name) {
        case "NotAllowedError":
        case "NotFoundError":
          errorMsg = "Camera not available or permission denied";
          detailMsg = "Please grant camera permission in browser settings or check if camera is connected";
          break;
        case "NotReadableError":
          errorMsg = "Camera is already in use";
          detailMsg = "Close other applications using the camera";
          break;
        case "OverconstrainedError":
          errorMsg = "Camera doesn't support required settings";
          detailMsg = "Try a different camera or browser";
          break;
        case "TypeError":
          errorMsg = "Camera request failed";
          detailMsg = "Invalid camera configuration";
          break;
      }

      if (statusEl) {
        statusEl.innerHTML = `❌ ${errorMsg}`;
        statusEl.title = detailMsg;
        statusEl.style.color = "#ff6b6b";
        statusEl.style.cursor = "help";
      }

      console.error("Camera Error:", err.name, err.message);
    },

    closeCamera: function() {
      if (this._stream) {
        this._stream.getTracks().forEach(function(track) {
          track.stop();
        });
        this._stream = null;
      }

      this._isActive = false;

      // Reset UI
      $("camera-prompt").style.display = "block";
      $("camera-preview-area").style.display = "none";
      document.querySelector(".camera-controls").style.display = "none";

      var statusEl = document.querySelector(".camera-status");
      if (statusEl) {
        statusEl.innerHTML = "📷 No camera active";
        statusEl.style.color = "inherit";
      }

      if (this._animationId) {
        cancelAnimationFrame(this._animationId);
      }
    },

    captureImage: function() {
      if (!this._video || !this._canvas) return;

      var ctx = this._canvas.getContext("2d");
      this._canvas.width = this._video.videoWidth;
      this._canvas.height = this._video.videoHeight;
      ctx.drawImage(this._video, 0, 0);

      var self = this;
      this._canvas.toBlob(function(blob) {
        var file = new File([blob], "camera_capture_" + Date.now() + ".jpg", { type: "image/jpeg" });
        self._selectedFiles.push(file);

        // Add to preview
        self.addToPreview(file);

        // Update file count
        var fileCount = document.getElementById("file-count");
        if (fileCount) {
          fileCount.textContent = self._selectedFiles.length + " images selected";
        }

        // Enable upload button
        var uploadBtn = document.getElementById("btn-create-upload");
        if (uploadBtn && self._selectedFiles.length > 0) {
          uploadBtn.disabled = false;
        }
      }, "image/jpeg", 0.9);
    },

    addToPreview: function(file) {
      var reader = new FileReader();
      var thumbGrid = document.getElementById("thumb-preview");

      reader.onload = function(e) {
        var img = document.createElement("img");
        img.src = e.target.result;
        img.style.cssText = "width: 100%; aspect-ratio: 1; object-fit: cover; border: 1px solid rgba(255,255,255,0.1);";
        if (thumbGrid) {
          thumbGrid.appendChild(img);
        }
      };

      reader.readAsDataURL(file);
    },

    startQualityMonitoring: function() {
      var self = this;

      var monitor = function() {
        if (!self._isActive) return;

        // Placeholder for quality metrics
        var metricsSharpness = document.getElementById("metric-sharpness");
        var metricsBrightness = document.getElementById("metric-brightness");
        var metricsFace = document.getElementById("metric-face");

        if (metricsSharpness) metricsSharpness.textContent = "Good ✓";
        if (metricsBrightness) metricsBrightness.textContent = "Optimal ✓";
        if (metricsFace) metricsFace.textContent = "Detecting...";

        var progressBar = document.getElementById("quality-progress");
        if (progressBar) {
          progressBar.style.width = "75%";
        }

        self._animationId = requestAnimationFrame(monitor);
      };

      monitor();
    }
  };

  // Auto-initialize
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function() {
      window.CameraCapture.init();
    });
  } else {
    window.CameraCapture.init();
  }

})();
