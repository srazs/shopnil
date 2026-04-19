import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np
import serial
#trashbot
# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
MODEL_PATH = "F:/AG/AI/idsqlast.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.6
SATURATION_FACTOR = 1  # Color boost

# ----- SERIAL CONFIG -----
SERIAL_PORT = "COM3"     # Change if needed com portr
BAUD_RATE = 115200
SEND_COOLDOWN = 1.5      # Safety gap between sends (sec)

# ----- VOTING WINDOW CONFIG -----
WINDOW_DURATION = 17.0   # seconds to accumulate detections

# ----------------------------------------
# ESP RESET HELPER
# ----------------------------------------
def reset_esp(s: serial.Serial):
    """
    Try to reset the ESP8266 using DTR/RTS lines.
    """
    try:
        print("[DEBUG] Resetting ESP8266 via DTR/RTS...")
        s.setDTR(False)
        s.setRTS(True)
        time.sleep(0.1)
        s.setRTS(False)
        time.sleep(0.3)  # give ESP time to boot
        s.reset_input_buffer()
        print("[DEBUG] ESP8266 reset sequence done.")
    except Exception as e:
        print(f"[DEBUG] Failed to reset ESP8266 via DTR/RTS: {e}")


# ----------------------------------------
# SERIAL INITIALIZATION
# ----------------------------------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    print(f"[DEBUG] 🔗 Serial connected to {SERIAL_PORT} at {BAUD_RATE}")
    reset_esp(ser)  # 🔁 reset ESP every time script starts
except Exception as e:
    ser = None
    print(f"[DEBUG] ❌ Serial connection failed: {e}")
    print("[DEBUG] Running detection only. No commands will be sent to ESP.")

last_sent_time = 0.0
last_esp_line = ""
last_esp_time = 0.0

# For auto-dump + auto-return behavior & overlay text
last_dump_time = 0.0
last_return_time = 0.0
DUMP_COOLDOWN = 5.0        # seconds between auto-dump triggers
RETURN_COOLDOWN = 5.0      # seconds between auto-return triggers

status_text = ""
status_text_time = 0.0
STATUS_TEXT_DURATION = 5.0  # seconds to show status text

# Detection state: True = bot at home / scanning, False = bot on mission
detection_enabled = True


def send_to_esp(command: str):
    """
    Send a command string to the AI ESP over serial.
      'organic'   -> sendCommandToServo("ORG")
      'inorganic' -> sendCommandToServo("INO")
      'go'        -> sendCommandToSensor(CMD_GO)
      'dump'      -> sendCommandToServo("DMP")
      'return'    -> sendCommandToSensor(CMD_RETURN)
    """
    global last_sent_time

    if ser is None:
        print(f"[DEBUG] Skipping send, no serial connection. Wanted to send: {command}")
        return

    now = time.time()
    if (now - last_sent_time) < SEND_COOLDOWN:
        print(f"[DEBUG] Cooldown active. Not sending '{command}'. "
              f"Time since last send: {now - last_sent_time:.2f}s")
        return

    try:
        msg = command + "\n"
        ser.write(msg.encode())
        print(f"[DEBUG] 📡 Sent to ESP: '{msg.strip()}'")
        last_sent_time = now
    except Exception as e:
        print(f"[DEBUG] ❌ Serial send error: {e}")


def read_from_esp():
    """
    Read any available messages from ESP and print them.
    Also:
      - 'Bot at destination' -> auto 'dump'
      - 'OK_DMP'            -> auto 'return'
      - 'Bot home'          -> resume detection ONLY if it was paused
    """
    global last_esp_line, last_esp_time
    global last_dump_time, last_return_time
    global status_text, status_text_time
    global detection_enabled, window_start_time, count_organic, count_inorganic

    if ser is None:
        return

    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            now = time.time()
            # Ignore exact duplicate within 0.5s
            if line == last_esp_line and (now - last_esp_time) < 0.5:
                continue

            last_esp_line = line
            last_esp_time = now
            print(f"[ESP → PC] {line}")

            # ---- AUTO-DUMP TRIGGER ----
            if "Bot at destination" in line:
                if (now - last_dump_time) > DUMP_COOLDOWN:
                    print("[DEBUG] Detected 'Bot at destination' from ESP. Sending 'dump' command.")
                    send_to_esp("dump")
                    last_dump_time = now
                    status_text = "Dumping trash..."
                    status_text_time = now

            # ---- AUTO-RETURN TRIGGER ----
            if "OK_DMP" in line:
                if (now - last_return_time) > RETURN_COOLDOWN:
                    print("[DEBUG] Detected 'OK_DMP' from ESP. Sending 'return' command.")
                    send_to_esp("return")
                    last_return_time = now
                    status_text = "Returning home..."
                    status_text_time = now

            # ---- BOT HOME -> RESUME DETECTION (ONLY IF WAS PAUSED) ----
            if "Bot home" in line:
                if not detection_enabled:
                    print("[DEBUG] Detected 'Bot home' from ESP while detection paused. "
                          "Resuming detection & resetting window.")
                    detection_enabled = True
                    # Reset voting state so new detections are fresh
                    window_start_time = now
                    count_organic = 0
                    count_inorganic = 0
                    status_text = "Bot home – resuming detection"
                    status_text_time = now
                else:
                    # Already in detection-enabled state, ignore periodic Bot home spam
                    print("[DEBUG] 'Bot home' received but detection already enabled; ignoring.")

    except Exception as e:
        print(f"[DEBUG] ❌ Error reading from ESP: {e}")


# ----------------------------------------
# LOAD MODEL
# ----------------------------------------
print(f"\n[DEBUG] 🚀 Loading model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print(f"[DEBUG] ✅ Model loaded on device: {DEVICE}")

# ----------------------------------------
# INITIALIZE CAMERA
# ----------------------------------------
print("[DEBUG] Initializing webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam. Check camera connection or index (0, 1, etc.)")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[DEBUG] Camera opened at {frame_width}x{frame_height}")

# ----------------------------------------
# ROI (Region of Interest)
# ----------------------------------------
roi_w = frame_width // 4
roi_h = frame_height // 2
x1 = (frame_width - roi_w) // 2
y1 = (frame_height - roi_h) // 2
x2 = x1 + roi_w
y2 = y1 + roi_h
print(f"[DEBUG] ROI defined: x1={x1}, y1={y1}, x2={x2}, y2={y2}")


# ----------------------------------------
# PREPROCESSING
# ----------------------------------------
def preprocess_frame(frame):
    # Light bilateral filter (denoising)
    filtered = cv2.bilateralFilter(frame, d=3, sigmaColor=20, sigmaSpace=20)

    # Saturation boost
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= SATURATION_FACTOR
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Slight brightness/contrast tweak
    enhanced = cv2.convertScaleAbs(saturated, alpha=1.05, beta=5)
    return enhanced


# ----------------------------------------
# VOTING STATE
# ----------------------------------------
window_start_time = time.time()
count_organic = 0
count_inorganic = 0

# ----------------------------------------
# LIVE DETECTION LOOP
# ----------------------------------------
prev_time = 0.0
print("\n[DEBUG] 🎥 Starting detection loop. Press 'Q' to quit. Press 'F' to send GO.\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[DEBUG] ⚠️ Failed to grab frame. Exiting loop.")
        break

    frame_count += 1

    # Draw ROI box (always)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    roi = frame[y1:y2, x1:x2]

    # ---------------------------------
    # DETECTION & VOTING (ONLY IF ENABLED)
    # ---------------------------------
    if detection_enabled:
        # Preprocess
        clean_roi = preprocess_frame(roi)

        # Run YOLO on ROI
        results = model(clean_roi, conf=CONFIDENCE_THRESHOLD, device=DEVICE, verbose=False)

        # Annotate ROI and put back onto main frame
        annotated_roi = results[0].plot()
        frame[y1:y2, x1:x2] = annotated_roi

        names = model.names
        detected_label = None

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                label = names[cls].lower()
                conf = float(box.conf[0].item())

                print(f"[DEBUG] Detected: {label} (conf={conf:.2f})")

                if label in ["organic", "inorganic"]:
                    detected_label = label
                    if detected_label == "organic":
                        count_organic += 1
                    elif detected_label == "inorganic":
                        count_inorganic += 1

                    print(f"[DEBUG] Voting counts -> organic: {count_organic}, inorganic: {count_inorganic}")
                    break

        # WINDOW CHECK only when detection is active
        now = time.time()
        elapsed_window = now - window_start_time

        if elapsed_window >= WINDOW_DURATION:
            print("\n[DEBUG] ===== WINDOW COMPLETE =====")
            print(f"[DEBUG] Final counts -> organic: {count_organic}, inorganic: {count_inorganic}")

            if count_organic == 0 and count_inorganic == 0:
                print("[DEBUG] No detections in this window. No command sent.")
            else:
                if count_organic > count_inorganic:
                    majority_label = "organic"
                elif count_inorganic > count_organic:
                    majority_label = "inorganic"
                else:
                    # Tie handling: default to inorganic (you can change this)
                    majority_label = "inorganic"
                    print("[DEBUG] Tie in counts, defaulting to 'inorganic'")

                print(f"[DEBUG] Majority decision for this window: {majority_label}")
                send_to_esp(majority_label)

            # Reset window + counters
            window_start_time = now
            count_organic = 0
            count_inorganic = 0
            print("[DEBUG] Counters reset. Starting new window.\n")

    else:
        # Detection paused: no YOLO, no voting, no window timer
        elapsed_window = 0.0

    # ---------------------------------
    # READ ANY MESSAGES FROM ESP
    # ---------------------------------
    read_from_esp()

    # FPS display
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # Overlays: FPS + voting state + detection state
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if detection_enabled:
        cv2.putText(frame, f"Window: {elapsed_window:5.1f}s / {WINDOW_DURATION:.0f}s",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Org: {count_organic}  Inorg: {count_inorganic}",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        det_state_str = "Detection: ON (bot at home)"
    else:
        cv2.putText(frame, "Detection paused (bot on mission)", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        det_state_str = "Detection: PAUSED"

    cv2.putText(frame, det_state_str, (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, "Press F to send GO", (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Status text overlay (e.g., "Dumping trash...", "Returning home...", "Bot home – resuming detection")
    if status_text and (time.time() - status_text_time) < STATUS_TEXT_DURATION:
        cv2.putText(frame, status_text, (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Organic vs Inorganic Detection (Mission-aware Controller)", frame)

    # ---------- KEY HANDLING ----------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("[DEBUG] 'Q' pressed. Exiting.")
        break

    if key == ord('f') or key == ord('F'):
        print("[DEBUG] 'F' pressed. Sending 'go' to ESP and pausing detection.")
        send_to_esp("go")
        # Pause detection and wipe any current voting pipeline state
        detection_enabled = False
        window_start_time = time.time()
        count_organic = 0
        count_inorganic = 0
        status_text = "Going to destination..."
        status_text_time = time.time()

# ----------------------------------------
# CLEANUP
# ----------------------------------------
cap.release()
cv2.destroyAllWindows()

if ser is not None:
    ser.close()
    print("[DEBUG] 🔌 Serial port closed.")

print("\n[DEBUG] 🛑 Detection stopped. Webcam released.")
3