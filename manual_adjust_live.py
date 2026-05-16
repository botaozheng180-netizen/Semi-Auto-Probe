"""
manual_adjust.py

Manual focusing assistant based on camera_test.py.

Purpose:
- Open the Litocam camera using the working camera_test backend.
- Capture frames one by one.
- Compute ROI-based brightness and focus/sharpness metrics.
- Guide the user through manual focusing screw/platform adjustment.
- Keep one OpenCV display window open and update it every iteration.
- Use a manually selected metal-pad/contact-edge ROI for MoS2 FET focusing.
- Track both the latest physical position and the best focus seen during each adjustment window.
- Save an image only when the latest physical position satisfies the target repeatedly, unless the user manually saves.

Usage:
1. Put this file in the same folder as camera_test.py.
2. Set MAGNIFICATION_LABEL to the current microscope magnification.
3. Close LitoDigital before running.
4. Run: python manual_adjust.py
5. Select a rectangular ROI around a visible metal-pad/contact edge when prompted.
6. Use the display window and terminal prompts to guide manual adjustment.

Key controls in the OpenCV window:
- q: quit
- s: save current latest frame manually
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

# Focus score targets calibrated from visually clear images.
# Raw Laplacian focus scores are magnification-dependent, so choose the current
# microscope magnification here before running the script.
MAGNIFICATION_LABEL = "20x"  # choose from: "5x", "10x", "20x", "50x", "100x"

FOCUS_TARGETS = {
    "5x": 875.0,
    "10x": 480.0,
    "20x": 210.0,
    "50x": 75.0,
    "100x": 60.0,
}

FOCUS_TARGET = FOCUS_TARGETS[MAGNIFICATION_LABEL]

# Require the latest physical position to satisfy the target repeatedly.
# This prevents saving one lucky/noisy frame.
CONSECUTIVE_GOOD_REQUIRED = 2

# Do not reset the consecutive-good counter after a single borderline/noisy frame.
# Example: good -> 1/2, one bad -> still 1/2, second bad -> reset to 0/2.
BAD_FRAMES_TO_RESET = 2

# Brightness limits. The sharpness score is less meaningful if the ROI is too dark
# or if the contact-edge region is too saturated.
MIN_ROI_MEAN_BRIGHTNESS = 10.0
MAX_ROI_MEAN_BRIGHTNESS = 240.0

# Whole-frame saturation is a warning only, because large metal pads can be bright.
FRAME_MAX_SATURATION_FRACTION = 0.20

# ROI/contact-edge saturation is used for acceptance/rejection.
EDGE_MAX_SATURATION_FRACTION = 0.20

# ROI settings
USE_CONTACT_EDGE_ROI = True
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
EDGE_DILATE_PIXELS = 4
MIN_EDGE_PIXELS = 50

# Display settings
MAX_DISPLAY_WIDTH = 1280
WINDOW_NAME = "Manual focus assistant"
ROI_SELECT_WINDOW = "Select metal-pad/contact-edge ROI -> Press Enter"

# Timing settings
READ_PROMPT_TIMEOUT = 5.0
ADJUSTMENT_TIMEOUT = 5.0
POST_CAPTURE_DELAY = 1.0
LIVE_CAPTURE_INTERVAL = 0.5

# Capture settings
WARMUP_FRAMES = 3
CAPTURE_TIMEOUT = 10.0

# Sensitivity settings for prompt wording.
PROMPT_REL_EPSILON = 0.005
PROMPT_ABS_EPSILON = 0.5


# =========================
# Data containers
# =========================

ROI = tuple[int, int, int, int]


@dataclass
class FrameMetrics:
    score: float
    mean: float
    min_value: int
    max_value: int
    saturation_fraction: float          # ROI/contact-edge saturation used for acceptance
    whole_saturation_fraction: float    # whole-frame saturation shown as warning only
    focus_ok: bool
    good_brightness: bool
    good_focus: bool
    metric_source: str


@dataclass
class AdjustmentResult:
    latest_frame: np.ndarray | None
    latest_metrics: FrameMetrics | None
    best_frame: np.ndarray | None
    best_metrics: FrameMetrics | None
    best_age_seconds: float | None
    action: str | None


# =========================
# ROI and image analysis helpers
# =========================

def valid_roi(roi: ROI | None) -> bool:
    """Return True if roi looks usable."""
    if roi is None:
        return False
    _x, _y, w, h = roi
    return w > 5 and h > 5


def crop_roi(frame: np.ndarray, roi: ROI | None) -> np.ndarray:
    """Crop a full-resolution frame to ROI. Falls back to whole frame."""
    if not valid_roi(roi):
        return frame

    x, y, w, h = roi
    height, width = frame.shape[:2]

    x0 = max(0, min(x, width - 1))
    y0 = max(0, min(y, height - 1))
    x1 = max(x0 + 1, min(x + w, width))
    y1 = max(y0 + 1, min(y + h, height))

    return frame[y0:y1, x0:x1]


def resize_for_display(frame: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH) -> np.ndarray:
    """Resize a frame for screen display without changing the saved full-res image."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame.copy()

    scale = max_width / width
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size)


def select_contact_roi(frame: np.ndarray) -> ROI | None:
    """
    Let the user select a rectangle around a visible metal-pad/contact edge.

    Good ROI choice:
    - contains part of a bright metal pad and part of the surrounding darker region
    - includes a crisp metal boundary/contact edge
    - avoids selecting only the flat metal-pad interior
    """
    display = resize_for_display(frame)

    scale_x = frame.shape[1] / display.shape[1]
    scale_y = frame.shape[0] / display.shape[0]

    print("\nSelect a rectangle around a visible metal-pad/contact edge.")
    print("Press Enter/Space to confirm, or Esc/c to cancel and use whole-frame scoring.")

    roi_display = cv2.selectROI(
        ROI_SELECT_WINDOW,
        display,
        fromCenter=False,
        showCrosshair=True,
    )

    try:
        cv2.destroyWindow(ROI_SELECT_WINDOW)
    except Exception:
        pass

    x, y, w, h = roi_display
    if w <= 0 or h <= 0:
        print("No ROI selected. Falling back to whole-frame scoring.")
        return None

    roi_full = (
        int(x * scale_x),
        int(y * scale_y),
        int(w * scale_x),
        int(h * scale_y),
    )
    print("Selected full-resolution ROI:", roi_full)
    return roi_full


def contact_edge_mask(roi_frame: np.ndarray) -> np.ndarray:
    """
    Detect a narrow edge band inside the selected ROI.

    This is used mostly for saturation checking near the metal-pad/contact boundary.
    The focus score itself is computed on the selected ROI, which is often more
    stable than computing only on sparse Canny pixels.
    """
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    kernel_size = 2 * EDGE_DILATE_PIXELS + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    edge_band = cv2.dilate(edges, kernel, iterations=1) > 0

    return edge_band


def focus_score(frame: np.ndarray, roi: ROI | None = None) -> float:
    """
    Compute raw ROI Laplacian-variance focus score.

    This is magnification-dependent, so use MAGNIFICATION_LABEL-specific targets.
    """
    target = crop_roi(frame, roi)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze_frame(frame: np.ndarray, roi: ROI | None = None) -> FrameMetrics:
    """Return focus and brightness metrics for one frame."""
    target = crop_roi(frame, roi)
    score = focus_score(frame, roi)
    mean = float(target.mean())
    min_value = int(target.min())
    max_value = int(target.max())

    # Pixels close to 255 are treated as saturated.
    whole_saturation_fraction = float(np.mean(frame >= 250))

    if valid_roi(roi):
        edge_band = contact_edge_mask(target)
        edge_pixel_count = int(np.count_nonzero(edge_band))

        if edge_pixel_count >= MIN_EDGE_PIXELS:
            edge_pixels = target[edge_band]
            saturation_fraction = float(np.mean(edge_pixels >= 250))
            metric_source = "ROI with edge mask"
        else:
            saturation_fraction = float(np.mean(target >= 250))
            metric_source = "ROI fallback: not enough edge pixels"
    else:
        saturation_fraction = whole_saturation_fraction
        metric_source = "Whole frame: no valid ROI"

    focus_ok = score >= FOCUS_TARGET
    brightness_in_range = MIN_ROI_MEAN_BRIGHTNESS <= mean <= MAX_ROI_MEAN_BRIGHTNESS
    edge_saturation_ok = saturation_fraction <= EDGE_MAX_SATURATION_FRACTION
    good_brightness = brightness_in_range and edge_saturation_ok
    good_focus = focus_ok and good_brightness

    return FrameMetrics(
        score=score,
        mean=mean,
        min_value=min_value,
        max_value=max_value,
        saturation_fraction=saturation_fraction,
        whole_saturation_fraction=whole_saturation_fraction,
        focus_ok=focus_ok,
        good_brightness=good_brightness,
        good_focus=good_focus,
        metric_source=metric_source,
    )


def score_change_threshold(reference_score: float | None) -> float:
    """Return the absolute score change needed to consider a trend meaningful."""
    if reference_score is None:
        return PROMPT_ABS_EPSILON
    return max(PROMPT_ABS_EPSILON, PROMPT_REL_EPSILON * abs(reference_score))


def make_prompt(previous_score: float | None, current_score: float) -> str:
    """Generate a manual focusing prompt from score trend."""
    if previous_score is None:
        return "Baseline captured. Adjust the focus screw/platform slightly."

    delta = current_score - previous_score
    threshold = score_change_threshold(previous_score)

    if abs(delta) < threshold:
        return "Focus changed only slightly. Continue slowly and watch for a peak."

    if delta > 0:
        return "Focus improved. Continue in the same direction, but use smaller steps."

    return "Focus worsened. You may have passed the focal point; reverse slightly."


def best_latest_prompt(best_metrics: FrameMetrics | None, latest_metrics: FrameMetrics) -> str:
    """
    Explain whether the latest physical position is near the best focus seen during
    the adjustment interval.
    """
    if best_metrics is None:
        return "No best-frame information this round."

    drop = best_metrics.score - latest_metrics.score
    threshold = score_change_threshold(best_metrics.score)

    if drop > threshold:
        return (
            f"Best this round was {best_metrics.score:.2f}, latest is {latest_metrics.score:.2f}. "
            "You likely passed the peak; reverse slightly."
        )

    return (
        f"Latest score {latest_metrics.score:.2f} is close to best this round "
        f"({best_metrics.score:.2f}). Continue slowly or stop."
    )


# =========================
# Display and saving helpers
# =========================

def draw_roi_rectangle(display: np.ndarray, frame_shape: tuple[int, int, int], roi: ROI | None) -> None:
    """Draw selected ROI rectangle on the resized display image in-place."""
    if not valid_roi(roi):
        return

    x, y, w, h = roi
    frame_h, frame_w = frame_shape[:2]
    display_h, display_w = display.shape[:2]

    sx = display_w / frame_w
    sy = display_h / frame_h

    p1 = (int(x * sx), int(y * sy))
    p2 = (int((x + w) * sx), int((y + h) * sy))

    cv2.rectangle(display, p1, p2, (0, 255, 255), 2)


def draw_overlay(
    frame: np.ndarray,
    metrics: FrameMetrics | None,
    prompt: str,
    good_count: int,
    bad_count: int,
    roi: ROI | None = None,
) -> np.ndarray:
    """Draw focus score and prompt on a preview image."""
    display = resize_for_display(frame)
    draw_roi_rectangle(display, frame.shape, roi)
    overlay = display.copy()

    if metrics is None:
        metric_lines = [
            f"Magnification: {MAGNIFICATION_LABEL}, target: {FOCUS_TARGET:.2f}",
            "Focus score: not measured yet",
            "ROI/edge saturation: not measured yet",
            f"Consecutive good frames: {good_count}/{CONSECUTIVE_GOOD_REQUIRED}",
            f"Bad frames toward reset: {bad_count}/{BAD_FRAMES_TO_RESET}",
        ]
    else:
        whole_sat_warning = metrics.whole_saturation_fraction > FRAME_MAX_SATURATION_FRACTION
        metric_lines = [
            f"Focus score: {metrics.score:.2f} / target {FOCUS_TARGET:.2f} ({MAGNIFICATION_LABEL})",
            f"Focus OK: {metrics.focus_ok}, brightness OK: {metrics.good_brightness}",
            f"Metric source: {metrics.metric_source}",
            f"ROI brightness: {metrics.mean:.1f}, min/max: {metrics.min_value}/{metrics.max_value}",
            f"ROI/edge saturation: {metrics.saturation_fraction:.4f}",
            f"Whole-frame saturation: {metrics.whole_saturation_fraction:.4f}, warning={whole_sat_warning}",
            f"Consecutive good frames: {good_count}/{CONSECUTIVE_GOOD_REQUIRED}",
            f"Bad frames toward reset: {bad_count}/{BAD_FRAMES_TO_RESET}",
        ]

    lines = metric_lines + [
        prompt,
        "Keys: q quit | s save current latest frame",
    ]

    x, y0 = 20, 30
    line_gap = 28

    cv2.rectangle(overlay, (10, 10), (1250, 10 + line_gap * len(lines)), (0, 0, 0), -1)
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
    bad_count: int,
    seconds: float,
    roi: ROI | None = None,
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
        display = draw_overlay(frame, metrics, countdown_prompt, good_count, bad_count, roi=roi)
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
    filename = f"{label}_{timestamp}_{MAGNIFICATION_LABEL}_score_{metrics.score:.1f}.png"
    path = OUTPUT_DIR / filename

    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for path: {path}")

    return path


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


def live_adjust_capture(
    hcam,
    width: int,
    height: int,
    previous_score: float | None,
    good_count: int,
    bad_count: int,
    roi: ROI | None = None,
) -> AdjustmentResult:
    """
    During the manual adjustment window, repeatedly capture frames.

    latest_* represents the current physical platform position at the end of the
    interval. best_* represents the sharpest frame seen during the interval.
    Since the focus screw is manual and has no encoder, the program uses latest
    for the current state and best only as guidance.
    """
    best_frame = None
    best_metrics = None
    best_time = None

    latest_frame = None
    latest_metrics = None

    start = time.time()
    next_capture_time = 0.0

    while True:
        elapsed = time.time() - start
        remaining = max(0.0, ADJUSTMENT_TIMEOUT - elapsed)

        if elapsed >= ADJUSTMENT_TIMEOUT:
            break

        now = time.time()

        if now >= next_capture_time:
            latest_frame = capture_frame(hcam, width, height)
            latest_metrics = analyze_frame(latest_frame, roi)

            # Prefer brightness-valid frames for best focus. If no valid frame exists
            # yet, keep the best available frame so the loop can continue.
            can_compete = latest_metrics.good_brightness or best_metrics is None
            if can_compete and (best_metrics is None or latest_metrics.score > best_metrics.score):
                best_frame = latest_frame
                best_metrics = latest_metrics
                best_time = elapsed

            next_capture_time = now + LIVE_CAPTURE_INTERVAL

        if latest_frame is not None and latest_metrics is not None:
            best_score = best_metrics.score if best_metrics is not None else latest_metrics.score
            if best_time is None:
                age_text = "unknown"
            else:
                age_text = f"{max(0.0, elapsed - best_time):.1f}s ago"

            trend_prompt = make_prompt(previous_score, latest_metrics.score)
            prompt = (
                f"Live: latest {latest_metrics.score:.2f}, best {best_score:.2f} "
                f"({age_text}). {trend_prompt} ({remaining:.1f}s)"
            )
            display = draw_overlay(latest_frame, latest_metrics, prompt, good_count, bad_count, roi=roi)
            cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            best_age = None if best_time is None else max(0.0, time.time() - start - best_time)
            return AdjustmentResult(latest_frame, latest_metrics, best_frame, best_metrics, best_age, "q")

        if key == ord("s") and latest_frame is not None and latest_metrics is not None:
            path = save_frame(latest_frame, latest_metrics, label="manual_save_latest")
            print("Manual save latest:", path)

    best_age = None if best_time is None else max(0.0, time.time() - start - best_time)
    return AdjustmentResult(latest_frame, latest_metrics, best_frame, best_metrics, best_age, None)


# =========================
# Main manual adjustment loop
# =========================

def main():
    hcam = None
    current_frame: np.ndarray | None = None
    current_metrics: FrameMetrics | None = None
    contact_roi: ROI | None = None

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

        if USE_CONTACT_EDGE_ROI:
            contact_roi = select_contact_roi(current_frame)

        current_metrics = analyze_frame(current_frame, contact_roi)

        print("Baseline focus score:", f"{current_metrics.score:.2f}")
        print("Baseline ROI brightness mean:", f"{current_metrics.mean:.2f}")
        print("Baseline ROI/edge saturation fraction:", f"{current_metrics.saturation_fraction:.4f}")
        print("Baseline frame saturation fraction:", f"{current_metrics.whole_saturation_fraction:.4f}")
        print("Baseline focus OK:", current_metrics.focus_ok)
        print("Baseline brightness OK:", current_metrics.good_brightness)
        print("Baseline metric source:", current_metrics.metric_source)

        previous_score: float | None = current_metrics.score
        good_count = 1 if current_metrics.good_focus else 0
        bad_count = 0

        if current_metrics.good_focus:
            path = save_frame(current_frame, current_metrics, label="baseline_good_focus")
            print("Baseline already meets target. Saved baseline image:", path)

        # Show the baseline results and give the user time to read.
        action = wait_with_preview(
            current_frame,
            current_metrics,
            "Baseline captured. Read the score and prompt.",
            good_count,
            bad_count,
            READ_PROMPT_TIMEOUT,
            roi=contact_roi,
        )
        if action == "q":
            return
        if action == "s" and current_metrics is not None:
            path = save_frame(current_frame, current_metrics, label="manual_save_baseline")
            print("Manual save baseline:", path)

        print("\nManual focusing loop started.")
        print("The OpenCV window is intended to stay open and update every iteration.")
        print("Use q in the image window or Ctrl+C in terminal to quit.")
        print("Latest score represents the current physical platform position.")
        print("Best score is only guidance, because the manual screw has no position encoder.")

        while True:
            # 1. Give time to read previous prompts.
            prompt = make_prompt(previous_score, current_metrics.score if current_metrics else 0.0)
            action = wait_with_preview(
                current_frame,
                current_metrics,
                "Read prompt: " + prompt,
                good_count,
                bad_count,
                READ_PROMPT_TIMEOUT,
                roi=contact_roi,
            )
            if action == "q":
                break
            if action == "s" and current_frame is not None and current_metrics is not None:
                path = save_frame(current_frame, current_metrics, label="manual_save_current")
                print("Manual save current:", path)

            # 2. Live feedback while physically adjusting the focus screw/platform.
            result = live_adjust_capture(
                hcam,
                width,
                height,
                previous_score,
                good_count,
                bad_count,
                roi=contact_roi,
            )

            if result.action == "q":
                break
            if result.latest_frame is None or result.latest_metrics is None:
                print("No latest frame captured during live adjustment. Retrying...")
                continue

            # 3. Capture a new frame and record the best focus after manual adjustments.
            new_frame = result.latest_frame
            new_metrics = result.latest_metrics

            trend_prompt = make_prompt(previous_score, new_metrics.score)
            peak_prompt = best_latest_prompt(result.best_metrics, new_metrics)
            if result.best_age_seconds is not None and result.best_metrics is not None:
                peak_prompt += f" Best was about {result.best_age_seconds:.1f}s before the end of the interval."
            prompt = trend_prompt + " " + peak_prompt

            print("\nLatest focus score:", f"{new_metrics.score:.2f}")
            if result.best_metrics is not None:
                print("Best focus score this round:", f"{result.best_metrics.score:.2f}")
                if result.best_age_seconds is not None:
                    print("Best was about", f"{result.best_age_seconds:.1f}s", "before interval end")
            print("ROI brightness mean:", f"{new_metrics.mean:.2f}")
            print("ROI/edge saturation fraction:", f"{new_metrics.saturation_fraction:.4f}")
            print("Frame saturation fraction:", f"{new_metrics.whole_saturation_fraction:.4f}")
            print("Focus OK:", new_metrics.focus_ok)
            print("Brightness OK:", new_metrics.good_brightness)
            print("Metric source:", new_metrics.metric_source)
            print("Prompt:", prompt)

            if new_metrics.good_focus:
                good_count += 1
                bad_count = 0
            else:
                bad_count += 1
                if bad_count >= BAD_FRAMES_TO_RESET:
                    good_count = 0

            # 4. Save only when the target focus quality is repeatedly reached.
            if good_count >= CONSECUTIVE_GOOD_REQUIRED:
                path = save_frame(new_frame, new_metrics, label="accepted_focus_latest")
                print("Target reached at latest physical position. Saved accepted image:", path)
                good_count = 0
                bad_count = 0

            previous_score = new_metrics.score
            current_frame = new_frame
            current_metrics = new_metrics

            # 5. Briefly show the new result before next read/adjust step.
            action = wait_with_preview(
                current_frame,
                current_metrics,
                "New latest result: " + prompt,
                good_count,
                bad_count,
                POST_CAPTURE_DELAY,
                roi=contact_roi,
            )
            if action == "q":
                break
            if action == "s" and current_frame is not None and current_metrics is not None:
                path = save_frame(current_frame, current_metrics, label="manual_save_current")
                print("Manual save current:", path)

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