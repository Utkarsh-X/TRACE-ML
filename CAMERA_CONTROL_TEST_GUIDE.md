# Camera Control Implementation - Test Guide

## Overview
This document provides step-by-step instructions to verify the new **frontend-driven camera control** system is working correctly.

---

## 📋 Pre-Test Checklist

- [ ] All modified files are syntactically correct (verified ✓)
- [ ] Backend Python environment is ready
- [ ] Frontend is served correctly
- [ ] Browser developer console is open for debugging
- [ ] System has webcam connected

---

## 🧪 Test Procedure

### **Test 1: Backend Initialization (No Auto-Start)**

**Objective**: Verify backend does NOT auto-start camera

**Steps**:
1. Start the service:
   ```bash
   trace-aml service run --live
   ```

2. **Expected Output**:
   ```
   ╭──────────────────────────────╮
   │ Live recognition ready ...   │
   │ Enable camera from the       │
   │ frontend UI to start...      │
   ╰──────────────────────────────╯
   ```

3. **What to Check**:
   - ✅ Message says "camera disabled"
   - ✅ Message says "Enable camera from the frontend UI"
   - ✅ Backend process is running (no camera stream)

**What Should NOT happen**:
   - ❌ No message about "webcam index 0" auto-started
   - ❌ No frames being captured yet

---

### **Test 2: Camera Status Endpoint**

**Objective**: Verify `/api/v1/camera/status` returns disabled state on startup

**Steps**:
1. With backend running, in another terminal:
   ```bash
   curl http://127.0.0.1:8080/api/v1/camera/status
   ```

2. **Expected Response**:
   ```json
   {
     "enabled": false,
     "camera_index": 0,
     "resolution": "1280x720",
     "fps": 30
   }
   ```

3. **What to Check**:
   - ✅ `enabled: false` (camera is disabled)
   - ✅ All other fields present and correct

---

### **Test 3: Frontend Page Load - Camera Disabled**

**Objective**: Verify frontend shows "Enable Camera" button on page load

**Steps**:
1. Open Firefox/Chrome
2. Navigate to: `http://127.0.0.1:8080/ui/live_ops/index.html`
3. Check the Live Ops page

**Expected UI State**:
   - ✅ **Button text**: "▶ ENABLE CAMERA" (blue button)
   - ✅ **Camera area**: Shows placeholder with "Camera feed offline"
   - ✅ **Detection overlay**: Hidden/disabled

**Browser Console Check**:
   - Open Dev Tools (F12) → Console tab
   - Should show logs like: `[TraceClient] Connected` or similar
   - No errors about camera

---

### **Test 4: Enable Camera via Button**

**Objective**: Verify clicking "Enable Camera" starts backend capture

**Steps**:
1. On Live Ops page, click the blue "▶ ENABLE CAMERA" button
2. Wait 1-2 seconds
3. Observe the UI and backend

**Expected Behavior**:

   **Frontend**:
   - ✅ Button text changes to: "⏹ DISABLE CAMERA" (red button)
   - ✅ Camera placeholder disappears
   - ✅ MJPEG feed appears showing live camera
   - ✅ Detection overlay canvas appears over feed

   **Backend** (terminal):
   - ✅ No errors in the log
   - ✅ Should see frame processing (if logs enabled)
   - ✅ May see FPS metrics in forensic panel

   **Backend Verification**:
   ```bash
   curl http://127.0.0.1:8080/api/v1/camera/status
   ```
   Should return: `"enabled": true`

---

### **Test 5: Verify Data Flow**

**Objective**: Confirm backend is processing frames when enabled

**Steps**:
1. Camera should be enabled (from Test 4)
2. Check the **Live Ops page**:
   - Look at detection overlay (colored boxes over faces)
   - Check **right panel timeline** for recent detections
   - Check **Entity count** in header (should be > 0 if there are known persons)

3. In terminal, check detections:
   ```bash
   curl http://127.0.0.1:8080/api/v1/live/snapshot | jq '.recent_detections'
   ```

**Expected**:
   - ✅ Detections are being created
   - ✅ Timeline shows recent events
   - ✅ Live overlay shows boxes on faces

---

### **Test 6: Disable Camera via Button**

**Objective**: Verify clicking "Disable Camera" stops backend capture

**Steps**:
1. On Live Ops page, click the red "⏹ DISABLE CAMERA" button
2. Wait 1-2 seconds
3. Observe the UI and backend

**Expected Behavior**:

   **Frontend**:
   - ✅ Button text changes back to: "▶ ENABLE CAMERA" (blue)
   - ✅ MJPEG feed disappears
   - ✅ Camera placeholder reappears ("Camera feed offline")
   - ✅ Detection overlay canvas hides

   **Backend** (terminal):
   - ✅ No errors in the log
   - ✅ Processing should stop smoothly
   - ✅ No crash or exceptions

   **Backend Verification**:
   ```bash
   curl http://127.0.0.1:8080/api/v1/camera/status
   ```
   Should return: `"enabled": false`

---

### **Test 7: Toggle Multiple Times**

**Objective**: Verify enable/disable can be toggled repeatedly

**Steps**:
1. Click "Enable Camera" → Wait 2 sec
2. Click "Disable Camera" → Wait 2 sec
3. Click "Enable Camera" → Wait 2 sec
4. Click "Disable Camera"
5. Repeat 2-3 times

**Expected**:
   - ✅ No errors in browser console
   - ✅ No errors in backend terminal
   - ✅ Backend never crashes
   - ✅ Camera status endpoint always returns correct state
   - ✅ UI always matches backend state

---

### **Test 8: Page Refresh - State Sync**

**Objective**: Verify frontend syncs with backend state on page reload

**Steps**:
1. Enable camera (button shows "Disable Camera")
2. Press **F5** to refresh page
3. Wait for page to load
4. Check the button

**Expected**:
   - ✅ After refresh, button should show "⏹ DISABLE CAMERA" (red)
   - ✅ Camera feed should be active
   - ✅ Detections should be displayed
   - ✅ No manual re-enabling needed

**Alternative Test** (disable scenario):
1. Disable camera (button shows "Enable Camera")
2. Press **F5** to refresh
3. Check the button

**Expected**:
   - ✅ After refresh, button should show "▶ ENABLE CAMERA" (blue)
   - ✅ Camera placeholder should be visible
   - ✅ No feed should be displaying

---

### **Test 9: Resource Usage Check**

**Objective**: Verify camera disabled = lower CPU usage

**Steps**:
1. Open system Task Manager / Resource Monitor
2. Note CPU usage with camera **disabled**
3. Enable camera and wait 5 seconds
4. Note CPU usage with camera **enabled**
5. Disable camera and wait 5 seconds
6. Note CPU usage with camera **disabled** again

**Expected**:
   - ✅ CPU usage is lower when camera is disabled
   - ✅ CPU usage increases when camera is enabled
   - ✅ CPU usage decreases again when disabled
   - ✅ Noticeable difference (at least 10-20% depending on system)

---

## 🔍 Troubleshooting

### **Issue**: Button doesn't change text after clicking

**Debug**:
```bash
# Check terminal browser console (F12 → Console)
# Look for errors mentioning "camera" or "API"

# Check backend responses
curl -X POST http://127.0.0.1:8080/api/v1/camera/enable
# Should return: {"status": "enabled", "message": "..."}
```

### **Issue**: Camera feed shows but says "offline"

**Debug**:
```bash
# Check MJPEG endpoint is working
curl http://127.0.0.1:8080/api/v1/live/mjpeg -I
# Should return: Content-Type: multipart/x-mixed-replace

# Check camera status
curl http://127.0.0.1:8080/api/v1/camera/status
# Should show: "enabled": true
```

### **Issue**: Backend crashes when enabling camera

**Debug**:
- Check if camera is already open in another application
- Check system has webcam: `ls /dev/video*` (Linux) or Device Manager (Windows)
- Check backend logs for detailed error message

### **Issue**: Camera works but detection overlay is empty

**Debug**:
- You may not have any known persons enrolled
- Enroll a test person first via Enrollment page
- Or run `trace-aml person capture`

---

## ✅ Sign-Off Checklist

After all tests pass, mark completion:

- [ ] Test 1: Backend shows "camera disabled" on startup
- [ ] Test 2: Status endpoint returns `enabled: false` initially
- [ ] Test 3: Frontend shows "Enable Camera" button on page load
- [ ] Test 4: Clicking button enables camera and shows feed
- [ ] Test 5: Backend processes frames (detections visible)
- [ ] Test 6: Clicking again disables camera and hides feed
- [ ] Test 7: Toggle multiple times works without errors
- [ ] Test 8: Page refresh syncs state correctly
- [ ] Test 9: CPU usage matches camera state

---

## 📊 Expected Behavior Summary

| Action | Backend State | Frontend Button | Feed Display | Processing |
|--------|---------------|-----------------|--------------|------------|
| Startup | Disabled | "Enable Camera" | Offline | Idle |
| Click Enable | Enabled | "Disable Camera" | Live (MJPEG) | Active |
| Click Disable | Disabled | "Enable Camera" | Offline | Idle |
| Refresh (when enabled) | Enabled | "Disable Camera" | Live (MJPEG) | Active |
| Refresh (when disabled) | Disabled | "Enable Camera" | Offline | Idle |

---

## 📝 Notes

- **No auto-start**: Backend will NEVER auto-start camera anymore
- **User control**: Only user action from frontend can enable/disable camera
- **Resource efficient**: No wasted CPU when camera is not needed
- **Thread-safe**: All state changes use locks to prevent race conditions
- **Graceful degradation**: If camera endpoints fail, UI shows errors in console

---

## 🚀 After Verification

Once all tests pass:

1. ✅ Implementation is production-ready
2. ✅ No further modifications needed
3. ✅ Can be merged to main branch
4. ✅ Update documentation with new camera control feature

---

**Created**: April 13, 2026  
**Status**: Ready for Testing
