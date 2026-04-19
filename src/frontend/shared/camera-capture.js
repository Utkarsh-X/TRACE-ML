/**
 * Camera Capture Component for Enrollment
 *
 * Provides real-time camera feed with quality feedback for image capture
 * during the person enrollment process.
 */

(function() {
  "use strict";

  function $(id) { return document.getElementById(id); }

  // Module: CameraCapture
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
      // Create camera section container if not exists
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
            <p class="text-[0.75rem] text-outline mb-3">📷 No camera active</p>
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
            ❌ Close Camera
          </button>
        </div>

        <div id="captured-images" class="captured-images mt-4">
          <!-- Will show thumbnails of captured images -->
        </div>
      `;

      // Insert before drop-zone
      var dropZone = $("drop-zone");
      if (dropZone && dropZone.parentNode) {
        dropZone.parentNode.insertBefore(cameraSection, dropZone);
      }

      this._video = $("camera-video");
      this._canvas = $("camera-canvas");
    },

    attachEventListeners: function() {
      var self = this;

      // Open camera button
      var btnOpenCamera = $("btn-open-camera");
      if (btnOpenCamera) {
        btnOpenCamera.addEventListener("click", function() {
          self.openCamera();
        });
      }

      // Close camera button
      var btnCloseCamera = $("btn-close-camera");
      if (btnCloseCamera) {
        btnCloseCamera.addEventListener("click", function() {
          self.closeCamera();
        });
      }

      // Capture image button
      var btnCapture = $("btn-capture-image");
      if (btnCapture) {
        btnCapture.addEventListener("click", function() {
          self.captureImage();
        });
      }
    },

    openCamera: function() {
      var self = this;

      navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      .then(function(stream) {
        self._stream = stream;
        self._video.srcObject = stream;
        self._isActive = true;

        // Show camera UI
        $("camera-prompt").style.display = "none";
        $("camera-preview-area").style.display = "block";
        document.querySelector(".camera-controls").style.display = "flex";

        // Start quality feedback loop
        self.startQualityMonitoring();
      })
      .catch(function(err) {
        if (window.TraceToast) {
          window.TraceToast.error("Camera Access Denied", err.message);
        } else {
          alert("Camera access denied:\n" + err.message);
        }
      });
    },

    closeCamera: function() {
      if (this._stream) {
        this._stream.getTracks().forEach(function(track) {
          track.stop();
        });
      }
      this._isActive = false;
      this._video.srcObject = null;

      // Hide camera UI
      $("camera-prompt").style.display = "block";
      $("camera-preview-area").style.display = "none";
      document.querySelector(".camera-controls").style.display = "none";

      // Clear current feedback
      $("quality-metrics").innerHTML = '';

      if (this._animationId) {
        cancelAnimationFrame(this._animationId);
      }
    },

    startQualityMonitoring: function() {
      var self = this;

      var monitor = function() {
        if (!self._isActive) return;

        // Get current frame
        var ctx = self._canvas.getContext("2d");
        self._canvas.width = self._video.videoWidth;
        self._canvas.height = self._video.videoHeight;
        ctx.drawImage(self._video, 0, 0);

        // Assess quality (mock - in real app would call API)
        var imageData = self._canvas.toDataURL("image/jpeg");

        // Simulate quality assessment (in real app would POST to /api/v1/quality/assess)
        var quality = {
          sharpness: 55 + Math.random() * 45,
          brightness: 100 + Math.random() * 100,
          face_detected: Math.random() > 0.3,
        };

        // Update UI
        var qualityLevel = (quality.sharpness / 100) * 100;
        var progressEl = $("quality-progress");
        if (progressEl) {
          progressEl.style.width = qualityLevel + "%";
          progressEl.style.background = qualityLevel > 70 ? "#66bb6a" : (qualityLevel > 50 ? "#ffa726" : "#ef5350");
        }

        // Update metrics
        document.getElementById("metric-sharpness").textContent = quality.sharpness.toFixed(0);
        document.getElementById("metric-brightness").textContent = quality.brightness.toFixed(0);
        document.getElementById("metric-face").textContent = quality.face_detected ? "✓ Yes" : "✗ No";

        self._animationId = requestAnimationFrame(monitor);
      };

      monitor();
    },

    captureImage: function() {
      if (!this._isActive) {
        if (window.TraceToast) {
          window.TraceToast.warning("Camera Inactive", "Enable camera before capturing.");
        } else {
          alert("Camera not active");
        }
        return;
      }

      var ctx = this._canvas.getContext("2d");
      this._canvas.width = this._video.videoWidth;
      this._canvas.height = this._video.videoHeight;
      ctx.drawImage(this._video, 0, 0);

      // Convert to blob
      var self = this;
      this._canvas.toBlob(function(blob) {
        // Create File object for upload list
        var timestamp = new Date().toISOString().slice(0, 19);
        var file = new File(
          [blob],
          "capture_" + timestamp + ".jpg",
          { type: "image/jpeg" }
        );

        // Add to selected files (use global _selectedFiles if available)
        if (typeof _selectedFiles !== "undefined") {
          _selectedFiles.push(file);
        }

        // Update captured images display
        self.displayCapturedImage(file, URL.createObjectURL(blob));

        // Update main thumbnail grid if available
        if (typeof renderThumbnails === "function") {
          renderThumbnails();
        }
        if (typeof updateCreateButton === "function") {
          updateCreateButton();
        }
      }, "image/jpeg", 0.95);
    },

    displayCapturedImage: function(file, url) {
      var container = $("captured-images");
      if (!container) return;

      if (container.innerHTML === "") {
        container.innerHTML = '<p class="text-[0.7rem] text-outline mb-2">Captured Images:</p>';
      }

      var thumbContainer = document.createElement("div");
      thumbContainer.style.cssText = "display: grid; grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 6px; margin-top: 8px;";

      if (!container.querySelector(".captured-thumb-grid")) {
        var grid = document.createElement("div");
        grid.className = "captured-thumb-grid";
        grid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 6px;";
        container.appendChild(grid);
      }

      var grid = container.querySelector(".captured-thumb-grid");
      var thumb = document.createElement("img");
      thumb.src = url;
      thumb.style.cssText = "width: 100%; aspect-ratio: 1; object-fit: cover; border: 1px solid rgba(255,255,255,0.1); border-radius: 3px; cursor: pointer;";
      thumb.title = file.name;

      grid.appendChild(thumb);
    },
  };

  // Initialize when DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function() {
      CameraCapture.init();
    });
  } else {
    CameraCapture.init();
  }

})();
