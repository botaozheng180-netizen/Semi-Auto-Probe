"""
manual_adjust.py

Manual focusing assistant based on camera_test.py.

Purpose:
- Open the Litocam camera using the working camera_test backend.
- Capture frames one by one.
- Compute brightness and focus/sharpness metrics.
- Guide the user through manual focusing screw/platform adjustment.
- Keep one OpenCV display window open and update it every iteration.
- Save the image only when the target focus quality is reached, unless the user manually saves.

Usage:
1. Put this file in the same folder as camera_test.py.
2. Close LitoDigital before running.
3. Run: python manual_adjust.py
4. Use the display window and terminal prompts to guide manual adjustment.

Key controls in the OpenCV window:
- q: quit
- s: save current frame manually
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from ctypes import c_int, byref

import cv2
import numpy as np

# Reuse the working camera/DLL code from camera_test.py.
# camera_test.py must be in the same folder as this file.
import camera_test as ct


# =========================
# User-adjustable settings
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "manual_focus_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# The absolute focus score depends on magnification, illumination, sample texture,
# camera exposure, and ROI. Start with a lower value, observe typical scores,
# then raise/lower this threshold.
FOCUS_TARGET = 120.0

# Require the score to be close to or above target for this many consecutive frames.
# This prevents saving one lucky/noisy frame.
CONSECUTIVE_GOOD_REQUIRED = 2

# Brightness sanity limits. If the image is too dark or too saturated,
# the sharpness score is less meaningful.
MIN_MEAN_BRIGHTNESS = 10.0
MAX_MEAN_BRIGHTNESS = 240.0
MAX_SATURATION_FRACTION = 0.02  # fraction of pixels at/near 255

# Display settings
MAX_DISPLAY_WIDTH = 1280
WINDOW_NAME = "Manual focus assistant"

#Timing settings
READ_PROMPT_TIMEOUT = 5.0
ADJUSTMENT_TIMEOUT = 5.0
POST_CAPTURE_DELAY = 1.0

# Capture settings
WARMUP_FRAMES = 3
CAPTURE_TIMEOUT = 10.0


# =========================
# Data containers
# =========================

@dataclass
class FrameMetrics:
    score: float
    mean: float
    min_value: int
    max_value: int
    saturation_fraction: float
    good_brightness: bool
    good_focus: bool


# =========================
# Image analysis helpers
# =========================

def focus_score(frame: np.ndarray) -> float:
    """
    Compute a focus/sharpness score.

    Uses variance of Laplacian, same principle as camera_test.sharpness_score().
    Higher usually means sharper edges and better focus.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze_frame(frame: np.ndarray) -> FrameMetrics:
    """Return focus and brightness metrics for one frame."""
    score = focus_score(frame)
    mean = float(frame.mean())
    min_value = int(frame.min())
    max_value = int(frame.max())

    # Pixels close to 255 are treated as saturated.
    saturation_fraction = float(np.mean(frame >= 250))

    good_brightness = (
        MIN_MEAN_BRIGHTNESS <= mean <= MAX_MEAN_BRIGHTNESS
        and saturation_fraction <= MAX_SATURATION_FRACTION
    )
    good_focus = score >= FOCUS_TARGET and good_brightness

    return FrameMetrics(
        score=score,
        mean=mean,
        min_value=min_value,
        max_value=max_value,
        saturation_fraction=saturation_fraction,
        good_brightness=good_brightness,
        good_focus=good_focus,
    )


def resize_for_display(frame: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH) -> np.ndarray:
    """Resize a frame for screen display without changing the saved full-res image."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size)


def draw_overlay(
    frame: np.ndarray,
    metrics: FrameMetrics | None,
    prompt: str,
    good_count: int,
) -> np.ndarray:
    """Draw focus score and prompt on a preview image."""
    display = resize_for_display(frame)
    overlay = display.copy()

    if metrics is None:
        metric_lines = [
            "Focus score: not measured yet",
            "Mean brightness: not measured yet",
            "Saturation fraction: not measured yet",
            f"Consecutive good frames: {good_count}/{CONSECUTIVE_GOOD_REQUIRED}",
        ]
    else:
        metric_lines = [
            f"Focus score: {metrics.score:.2f} / target {FOCUS_TARGET:.2f}",
            f"Mean brightness: {metrics.mean:.1f}, min/max: {metrics.min_value}/{metrics.max_value}",
            f"Saturation fraction: {metrics.saturation_fraction:.4f}",
            f"Brightness OK: {metrics.good_brightness}",
            f"Consecutive good frames: {good_count}/{CONSECUTIVE_GOOD_REQUIRED}",
        ]
    lines = metric_lines + [
        prompt,
        "Keys: q quit | s save current frame | Enter/Space next capture",
    ]

    x, y0 = 20, 30
    line_gap = 28

    # Draw a dark background rectangle for readable text.
    cv2.rectangle(overlay, (10, 10), (950, 10 + line_gap * len(lines)), (0, 0, 0), -1)

    # Blend overlay with display.
    display = cv2.addWeighted(overlay, 0.55, display, 0.45, 0)

    for i, line in enumerate(lines):
        y = y0 + i * line_gap
        cv2.putText(
            display,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return display


def wait_with_preview(
        frame: np.ndarray,
        metrics: FrameMetrics | None,
        prompt: str,
        good_count: int,
        seconds: float,
) -> str | None:
    """
    Keep the OpenCV window responsive while the user reads/adjusts.

    Returns:
    - "q" if user presses q
    - "s" if user presses s
    - None otherwise
    """
    start = time.time()
    while True:
        elapsed = time.time() - start
        remaining = max(0.0, seconds - elapsed)

        countdown_prompt = f"{prompt}  ({remaining:.1f}s)"
        display = draw_overlay(frame, metrics, countdown_prompt, good_count)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            return "q"
        if key == ord("s"):
            return "s"

        if elapsed >= seconds:
            return None


def save_frame(frame: np.ndarray, metrics: FrameMetrics, label: str = "accepted") -> Path:
    """Save a full-resolution frame with score information in the filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}_score_{metrics.score:.1f}.png"
    path = OUTPUT_DIR / filename

    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for path: {path}")

    return path


def make_prompt(previous_score: float | None, current_score: float, last_move: str | None = None) -> str:
    """Generate a manual focusing prompt from score trend."""
    if previous_score is None:
        return "Initial frame captured. Adjust focus slightly, then capture again."

    delta = current_score - previous_score
    relative = delta / max(abs(previous_score), 1.0)

    if abs(relative) < 0.03:
        return "Focus score changed little. Try a smaller adjustment or improve illumination."

    if delta > 0:
        return "Focus improved. Continue in the same direction for the next adjustment."

    return "Focus worsened. Reverse the direction for the next adjustment."


# =========================
# Camera session helpers
# =========================

def open_camera():
    """Open the first detected Litocam camera and return handle, width, height."""
    devices = (ct.LitocamDeviceV2 * ct.MAX_CAMERAS)()
    count = ct.cam.Litocam_EnumV2(devices)

    print("Number of cameras found:", count)
    if count == 0:
        raise RuntimeError("No Litocam camera found.")

    for i in range(count):
        print(f"Camera {i}: {devices[i].displayname}, ID = {devices[i].id}")

    hcam = ct.cam.Litocam_Open(devices[0].id)
    if not hcam:
        raise RuntimeError("Litocam_Open failed. Make sure LitoDigital is closed.")

    width = c_int()
    height = c_int()
    hr = ct.cam.Litocam_get_Size(hcam, byref(width), byref(height))
    ct.check_hr(hr, "Litocam_get_Size")

    print("Opened camera:", devices[0].displayname)
    print("Image size:", width.value, "x", height.value)

    return hcam, width.value, height.value


def capture_frame(hcam, width: int, height: int) -> np.ndarray:
    """Pull one frame from the camera and convert it to an OpenCV image."""
    buffer, frame_w, frame_h, row_pitch = ct.pull_one_frame(
        hcam,
        width=width,
        height=height,
        bits=ct.BITS,
        timeout=CAPTURE_TIMEOUT,
    )
    return ct.buffer_to_frame(buffer, frame_w, frame_h, row_pitch)


# =========================
# Main manual adjustment loop
# =========================

def main():
    hcam = None
    current_frame: np.ndarray | None = None
    current_metrics: FrameMetrics | None = None

    try:
        hcam, width, height = open_camera()

        # Use camera_test's explicit exposure/gain configuration.
        ct.configure_camera(hcam)

        hr = ct.cam.Litocam_StartPullModeWithCallback(
            hcam,
            ct.event_callback,
            None,
        )
        ct.check_hr(hr, "Litocam_StartPullModeWithCallback")

        print("Warming up camera...")
        for i in range(WARMUP_FRAMES):
            print(f"Discarding warm-up frame {i + 1}/{WARMUP_FRAMES}")
            capture_frame(hcam, width, height)

        print("\nCapturing baseline frame before any manual adjustment...")
        current_frame = capture_frame(hcam, width, height)
        current_metrics = analyze_frame(current_frame)

        print("Baseline focus score:", f"{current_metrics.score:.2f}")
        print("Baseline brightness mean:", f"{current_metrics.mean:.2f}")
        print("Baseline saturation fraction:", f"{current_metrics.saturation_fraction:.4f}")

        previous_score: float | None = current_metrics.score
        good_count = 1 if current_metrics.good_focus else 0

        if current_metrics.good_focus:
            path = save_frame(current_frame, current_metrics, label="baseline_good_focus")
            print("Baseline already meets target. Saved baseline image:", path)

        # Show baseline and give user time to read it.
        action = wait_with_preview(
            current_frame,
            current_metrics,
            "Baseline captured. Read the score and prompt.",
            good_count,
            READ_PROMPT_TIMEOUT,
        )
        if action == "q":
            return
        if action == "s" and current_metrics is not None:
            path = save_frame(current_frame, current_metrics, label="manual_save")
            print("Manual save:", path)

        print("\nManual focusing loop started.")
        print("The OpenCV window is intended to stay open and update every iteration.")
        print("Do not close it using the X button during the wait; press q in the image window or Ctrl+C in terminal.")

        while True:
            # 1. Give time to read previous prompt.
            prompt = make_prompt(previous_score, current_metrics.score if current_metrics else 0.0)
            action = wait_with_preview(
                current_frame,
                current_metrics,
                "Read prompt: " + prompt,
                good_count,
                READ_PROMPT_TIMEOUT,
            )
            if action == "q":
                break
            if action == "s" and current_metrics is not None:
                path = save_frame(current_frame, current_metrics, label="manual_save")
                print("Manual save:", path)

           # 2. Give time to physically adjust focus screw/platform.
            action = wait_with_preview(
                current_frame,
                current_metrics,
                "Adjust focus screw/platform now.",
                good_count,
                ADJUSTMENT_TIMEOUT,
            )
            if action == "q":
                break
            if action == "s" and current_metrics is not None:
                path = save_frame(current_frame, current_metrics, label="manual_save")
                print("Manual save:", path)

            # 3. Capture a new frame after manual adjustment.
            new_frame = capture_frame(hcam, width, height)
            new_metrics = analyze_frame(new_frame)

            prompt = make_prompt(previous_score, new_metrics.score)

            print("\nFocus score:", f"{new_metrics.score:.2f}")
            print("Brightness mean:", f"{new_metrics.mean:.2f}")
            print("Saturation fraction:", f"{new_metrics.saturation_fraction:.4f}")
            print("Prompt:", prompt)

            if new_metrics.good_focus:
                good_count += 1
            else:
                good_count = 0

            # 4. Save only when the target focus quality is repeatedly reached.
            if good_count >= CONSECUTIVE_GOOD_REQUIRED:
                path = save_frame(new_frame, new_metrics, label="accepted_focus")
                print("Target reached. Saved accepted image:", path)
                good_count = 0

            previous_score = new_metrics.score
            current_frame = new_frame
            current_metrics = new_metrics

            # 5. Briefly show the new result before next read/adjust cycle.
            action = wait_with_preview(
                current_frame,
                current_metrics,
                "New result captured: " + prompt,
                good_count,
                POST_CAPTURE_DELAY,
            )
            if action == "q":
                break
            if action == "s":
                path = save_frame(current_frame, current_metrics, label="manual_save")
                print("Manual save:", path)
    finally:
        if hcam:
            try:
                ct.cam.Litocam_Stop(hcam)
            except Exception:
                pass
            ct.cam.Litocam_Close(hcam)

        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()