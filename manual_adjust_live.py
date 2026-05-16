"""
live_focus_litocam.py

Live focus feedback for the Litocam CMOS sensor.

This version does NOT use cv2.VideoCapture(), because that opens normal webcams.
Instead, it reuses the working Litocam SDK backend from camera_test.py.

Features:
- Opens the Litocam CMOS camera through litocam.dll
- Captures frames using SDK pull mode
- Optional ROI selection around metal-pad/contact edge
- Computes ROI-based Laplacian focus score
- Shows live video
- Draws a small rolling focus-score chart inside the display window
- Uses magnification-specific focus thresholds
- Uses EMA smoothing and hysteresis for more stable CLEAR/ADJUSTING feedback

Required:
- camera_test.py in the same folder
- LitoDigital closed before running
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import time
from collections import deque
from ctypes import byref, c_int
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Import the working Litocam backend
# ============================================================

def load_camera_test_backend():
    """
    Prefer normal import:
        import camera_test as ct

    Fallback:
        load camera_test(5).py if that is the local filename.
    """
    try:
        import camera_test as ct
        return ct
    except ImportError:
        script_dir = Path(__file__).resolve().parent
        fallback_path = script_dir / "camera_test(5).py"

        if not fallback_path.exists():
            raise ImportError(
                "Could not import camera_test.py. "
                "Put this file in the same folder as camera_test.py, "
                "or rename camera_test(5).py to camera_test.py."
            )

        spec = importlib.util.spec_from_file_location("camera_test_fallback", fallback_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load backend from {fallback_path}")

        ct = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ct)
        return ct


ct = load_camera_test_backend()


# ============================================================
# User-adjustable settings
# ============================================================

FOCUS_TARGETS = {
    "5x": 875.0,
    "10x": 480.0,
    "20x": 210.0,
    "50x": 75.0,
    "100x": 60.0,
}

MAX_DISPLAY_WIDTH = 1280
WINDOW_NAME = "Live Litocam Focus Feedback"
ROI_SELECT_WINDOW = "Select ROI -> Press Enter"

WARMUP_FRAMES = 3
CAPTURE_TIMEOUT = 10.0

CONSECUTIVE_GOOD_REQUIRED = 2
BAD_FRAMES_TO_RESET = 2

MIN_ROI_MEAN_BRIGHTNESS = 10.0
MAX_ROI_MEAN_BRIGHTNESS = 240.0
EDGE_MAX_SATURATION_FRACTION = 0.20
FRAME_MAX_SATURATION_FRACTION = 0.20

CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
EDGE_DILATE_PIXELS = 4
MIN_EDGE_PIXELS = 50

PROMPT_REL_EPSILON = 0.005
PROMPT_ABS_EPSILON = 0.5


# ============================================================
# Quiet Litocam callback and frame capture
# ============================================================

@ct.CALLBACK_TYPE
def quiet_event_callback(nEvent, pCallbackCtx):
    """
    Same logic as camera_test.event_callback, but without printing every event.
    Printing every callback would flood the terminal during live video.
    """
    if nEvent == ct.EVENT_IMAGE:
        ct.image_ready.set()


def open_litocam(camera_number: int = 0):
    """
    Open a Litocam CMOS camera through the SDK.

    camera_number refers to the index among Litocam SDK devices,
    not cv2.VideoCapture webcam indices.
    """
    devices = (ct.LitocamDeviceV2 * ct.MAX_CAMERAS)()
    count = ct.cam.Litocam_EnumV2(devices)

    print("Number of Litocam cameras found:", count)

    if count == 0:
        raise RuntimeError("No Litocam camera found. Close LitoDigital and check the USB connection.")

    for i in range(count):
        print(f"Camera {i}: {devices[i].displayname}, ID = {devices[i].id}")

    if camera_number < 0 or camera_number >= count:
        raise ValueError(f"Invalid Litocam camera number {camera_number}. Available: 0 to {count - 1}")

    hcam = ct.cam.Litocam_Open(devices[camera_number].id)

    if not hcam:
        raise RuntimeError("Litocam_Open failed. Make sure LitoDigital is closed.")

    width = c_int()
    height = c_int()

    hr = ct.cam.Litocam_get_Size(hcam, byref(width), byref(height))
    ct.check_hr(hr, "Litocam_get_Size")

    print("Opened Litocam:", devices[camera_number].displayname)
    print("Image size:", width.value, "x", height.value)

    return hcam, width.value, height.value


def start_litocam_stream(hcam):
    """
    Configure camera exposure/gain and start SDK pull mode.
    """
    ct.configure_camera(hcam)

    ct.image_ready.clear()

    hr = ct.cam.Litocam_StartPullModeWithCallback(
        hcam,
        quiet_event_callback,
        None,
    )
    ct.check_hr(hr, "Litocam_StartPullModeWithCallback")


def quiet_pull_one_frame(hcam, width: int, height: int, timeout: float = CAPTURE_TIMEOUT) -> np.ndarray:
    """
    Pull one frame from the Litocam SDK without verbose printing.

    This is a quieter version of camera_test.pull_one_frame().
    """
    bits = ct.BITS
    row_pitch = ct.calc_row_pitch(width, bits)
    buffer_size = row_pitch * height
    buffer = (ctypes.c_ubyte * buffer_size)()

    deadline = time.time() + timeout

    while time.time() < deadline:
        ct.image_ready.wait(timeout=0.5)
        ct.image_ready.clear()

        for _attempt in range(10):
            info = ct.LitocamFrameInfoV2()

            hr = ct.cam.Litocam_PullImageWithRowPitchV2(
                hcam,
                buffer,
                bits,
                row_pitch,
                byref(info),
            )

            code = ct.hr32(hr)

            if code == ct.S_OK:
                return ct.buffer_to_frame(buffer, info.width, info.height, row_pitch)

            if code == ct.E_PENDING:
                time.sleep(0.01)
                continue

            raise RuntimeError(
                f"Litocam_PullImageWithRowPitchV2 failed, HRESULT = {hex(code)}"
            )

    raise RuntimeError("Timed out waiting for a usable Litocam frame.")


def close_litocam(hcam):
    """
    Stop and close camera safely.
    """
    if hcam:
        try:
            ct.cam.Litocam_Stop(hcam)
        except Exception:
            pass

        try:
            ct.cam.Litocam_Close(hcam)
        except Exception:
            pass


# ============================================================
# ROI and image analysis helpers
# ============================================================

def valid_roi(roi):
    if roi is None:
        return False

    _x, _y, w, h = roi
    return w > 5 and h > 5


def resize_for_display(frame: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]

    if w <= max_width:
        return frame.copy()

    scale = max_width / w
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size)


def crop_roi(frame: np.ndarray, roi):
    if not valid_roi(roi):
        return frame

    x, y, w, h = roi
    frame_h, frame_w = frame.shape[:2]

    x0 = max(0, min(x, frame_w - 1))
    y0 = max(0, min(y, frame_h - 1))
    x1 = max(x0 + 1, min(x + w, frame_w))
    y1 = max(y0 + 1, min(y + h, frame_h))

    return frame[y0:y1, x0:x1]


def select_contact_roi(frame: np.ndarray):
    """
    Select ROI on a resized preview, then map it back to full-resolution coordinates.
    """
    display = resize_for_display(frame)

    scale_x = frame.shape[1] / display.shape[1]
    scale_y = frame.shape[0] / display.shape[0]

    print("\nSelect a rectangular ROI around a visible metal-pad/contact edge.")
    print("Press Enter/Space to confirm. Press Esc/c to cancel and use whole-frame scoring.")

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
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    kernel_size = 2 * EDGE_DILATE_PIXELS + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    edge_band = cv2.dilate(edges, kernel, iterations=1) > 0
    return edge_band


def calculate_focus_score(frame: np.ndarray, roi=None) -> float:
    target = crop_roi(frame, roi)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze_frame(frame: np.ndarray, roi, focus_target: float):
    target = crop_roi(frame, roi)

    score = calculate_focus_score(frame, roi)
    mean = float(target.mean())
    min_value = int(target.min())
    max_value = int(target.max())

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
            metric_source = "ROI fallback"
    else:
        saturation_fraction = whole_saturation_fraction
        metric_source = "Whole frame"

    focus_ok = score >= focus_target
    brightness_ok = (
        MIN_ROI_MEAN_BRIGHTNESS <= mean <= MAX_ROI_MEAN_BRIGHTNESS
        and saturation_fraction <= EDGE_MAX_SATURATION_FRACTION
    )

    good_focus = focus_ok and brightness_ok

    return {
        "score": score,
        "mean": mean,
        "min": min_value,
        "max": max_value,
        "saturation_fraction": saturation_fraction,
        "whole_saturation_fraction": whole_saturation_fraction,
        "focus_ok": focus_ok,
        "brightness_ok": brightness_ok,
        "good_focus": good_focus,
        "metric_source": metric_source,
    }


# ============================================================
# Focus stability and prompt logic
# ============================================================

def update_ema(previous_ema, current_value, alpha=0.35):
    if previous_ema is None:
        return current_value

    return alpha * current_value + (1.0 - alpha) * previous_ema


def score_change_threshold(reference_score: float | None) -> float:
    if reference_score is None:
        return PROMPT_ABS_EPSILON

    return max(PROMPT_ABS_EPSILON, PROMPT_REL_EPSILON * abs(reference_score))


def make_prompt(previous_score: float | None, current_score: float) -> str:
    if previous_score is None:
        return "Baseline captured. Adjust slowly."

    delta = current_score - previous_score
    threshold = score_change_threshold(previous_score)

    if abs(delta) < threshold:
        return "Focus changed only slightly. Continue slowly and watch for a peak."

    if delta > 0:
        return "Focus improved. Continue in the same direction with smaller steps."

    return "Focus worsened. You may have passed the focal point; reverse slightly."


class ConsecutiveFocusTracker:
    """
    Logic inherited from the manual-adjust version:

    - Need repeated good frames before saying CLEAR.
    - Do not reset immediately after one borderline bad frame.
    """

    def __init__(self, required_good=2, bad_to_reset=2):
        self.required_good = required_good
        self.bad_to_reset = bad_to_reset
        self.good_count = 0
        self.bad_count = 0

    def update(self, good_focus: bool):
        if good_focus:
            self.good_count = min(self.good_count + 1, self.required_good)
            self.bad_count = 0
        else:
            self.bad_count += 1

            if self.bad_count >= self.bad_to_reset:
                self.good_count = 0

        is_clear = self.good_count >= self.required_good
        return self.good_count, self.bad_count, is_clear


# ============================================================
# Drawing helpers
# ============================================================

def draw_roi_rectangle(display: np.ndarray, full_frame_shape, roi):
    if not valid_roi(roi):
        return

    x, y, w, h = roi

    full_h, full_w = full_frame_shape[:2]
    display_h, display_w = display.shape[:2]

    sx = display_w / full_w
    sy = display_h / full_h

    p1 = (int(x * sx), int(y * sy))
    p2 = (int((x + w) * sx), int((y + h) * sy))

    cv2.rectangle(display, p1, p2, (0, 255, 255), 2)


def draw_focus_chart(
    display: np.ndarray,
    scores,
    threshold: float,
    is_clear: bool = False,
    x=20,
    y=115,
    w=380,
    h=150,
):
    if len(scores) < 2:
        return display

    overlay = display.copy()

    cv2.rectangle(overlay, (x, y), (x + w, y + h), (245, 245, 245), -1)
    display = cv2.addWeighted(overlay, 0.75, display, 0.25, 0)

    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 0), 1)

    arr = np.array(scores, dtype=np.float32)

    min_score = float(np.min(arr))
    max_score = float(np.max(arr))

    min_score = min(min_score, threshold)
    max_score = max(max_score, threshold)

    if max_score - min_score < 1e-6:
        max_score = min_score + 1.0

    def score_to_y(score):
        return int(y + h - (score - min_score) / (max_score - min_score) * h)

    # Threshold line
    threshold_y = score_to_y(threshold)
    cv2.line(display, (x, threshold_y), (x + w, threshold_y), (0, 150, 0), 1)

    cv2.putText(
        display,
        f"Target {threshold:.0f}",
        (x + 8, max(y + 15, threshold_y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 100, 0),
        1,
        cv2.LINE_AA,
    )

    points = []
    for i, score in enumerate(arr):
        px = int(x + i * w / (len(arr) - 1))
        py = score_to_y(score)
        points.append((px, py))

    # Red when adjusting, green when clear
    line_color = (0, 180, 0) if is_clear else (0, 0, 255)

    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(display, p1, p2, line_color, 2)

    cv2.putText(
        display,
        f"Rolling focus: {scores[-1]:.1f}",
        (x + 8, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return display


def draw_dashboard(
    frame: np.ndarray,
    roi,
    metrics,
    ema_score: float,
    raw_score: float,
    focus_target: float,
    magnification: str,
    good_count: int,
    bad_count: int,
    is_clear: bool,
    best_score: float,
    previous_ema_score: float | None,
    scores,
):
    display = resize_for_display(frame)
    draw_roi_rectangle(display, frame.shape, roi)

    prompt = make_prompt(previous_ema_score, ema_score)

    whole_sat_warning = metrics["whole_saturation_fraction"] > FRAME_MAX_SATURATION_FRACTION
    status_text = "CLEAR" if is_clear else "ADJUSTING"
    status_color = (0, 180, 0) if is_clear else (0, 0, 255)

    lines = [
        f"{magnification} | EMA focus: {ema_score:.2f} | Raw: {raw_score:.2f} | Target: {focus_target:.2f}",
        f"Status: {status_text} | Good: {good_count}/{CONSECUTIVE_GOOD_REQUIRED} | Bad: {bad_count}",
        f"Best score this run: {best_score:.2f}",
        f"Brightness OK: {metrics['brightness_ok']} | Focus OK: {metrics['focus_ok']} | Source: {metrics['metric_source']}",
        f"ROI mean: {metrics['mean']:.1f}, min/max: {metrics['min']}/{metrics['max']}",
        f"ROI/edge saturation: {metrics['saturation_fraction']:.4f}",
        f"Whole-frame saturation: {metrics['whole_saturation_fraction']:.4f}, warning={whole_sat_warning}",
        prompt,
        "Keys: q quit | r reset best score",
    ]

    x, y0 = 20, 30
    line_gap = 25

    overlay = display.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (min(display.shape[1] - 10, 1250), 10 + line_gap * len(lines)),
        (0, 0, 0),
        -1,
    )
    display = cv2.addWeighted(overlay, 0.55, display, 0.45, 0)

    for i, line in enumerate(lines):
        y = y0 + i * line_gap

        color = status_color if i == 1 else (255, 255, 255)

        cv2.putText(
            display,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    display = draw_focus_chart(
        display,
        scores = scores,
        threshold = focus_target,
        is_clear = is_clear,
        x = 20,
        y = 10 + line_gap * len(lines) + 15,
        w = 400,
        h = 155,
    )

    return display


# ============================================================
# Main live loop
# ============================================================

def run_live_focus_litocam(
    camera_number: int = 0,
    magnification: str = "20x",
    target_fps: float = 10.0,
    use_roi: bool = True,
):
    if magnification not in FOCUS_TARGETS:
        raise ValueError(f"Unsupported magnification {magnification}. Choose from {list(FOCUS_TARGETS)}")

    focus_target = FOCUS_TARGETS[magnification]

    hcam = None

    try:
        hcam, width, height = open_litocam(camera_number)
        start_litocam_stream(hcam)

        print("\nWarming up camera...")
        for i in range(WARMUP_FRAMES):
            _ = quiet_pull_one_frame(hcam, width, height)
            print(f"Discarded warm-up frame {i + 1}/{WARMUP_FRAMES}")

        print("\nCapturing first frame...")
        first_frame = quiet_pull_one_frame(hcam, width, height)

        if use_roi:
            roi = select_contact_roi(first_frame)
        else:
            roi = None

        scores = deque(maxlen=120)
        tracker = ConsecutiveFocusTracker(
            required_good=CONSECUTIVE_GOOD_REQUIRED,
            bad_to_reset=BAD_FRAMES_TO_RESET,
        )

        ema_score = None
        previous_ema_score = None
        best_score = -1.0

        dt = 1.0 / target_fps if target_fps > 0 else 0.0

        print("\nLive Litocam focus feedback started.")
        print("Press q in the OpenCV window to quit.")
        print("Press r to reset best score.")
        print(f"Magnification: {magnification}")
        print(f"Focus target: {focus_target:.2f}")

        while True:
            loop_start = time.perf_counter()

            frame = quiet_pull_one_frame(hcam, width, height)

            raw_score = calculate_focus_score(frame, roi)
            previous_ema_score = ema_score
            ema_score = update_ema(ema_score, raw_score, alpha=0.35)

            metrics = analyze_frame(frame, roi, focus_target)

            # Use smoothed score for focus decision, but keep brightness decision from raw frame.
            smoothed_focus_ok = ema_score >= focus_target
            good_focus = smoothed_focus_ok and metrics["brightness_ok"]

            good_count, bad_count, is_clear = tracker.update(good_focus)

            if ema_score > best_score:
                best_score = ema_score

            scores.append(ema_score)

            display = draw_dashboard(
                frame=frame,
                roi=roi,
                metrics=metrics,
                ema_score=ema_score,
                raw_score=raw_score,
                focus_target=focus_target,
                magnification=magnification,
                good_count=good_count,
                bad_count=bad_count,
                is_clear=is_clear,
                best_score=best_score,
                previous_ema_score=previous_ema_score,
                scores=scores,
            )

            cv2.imshow(WINDOW_NAME, display)

            elapsed = time.perf_counter() - loop_start
            delay_ms = max(1, int((dt - elapsed) * 1000))

            key = cv2.waitKey(delay_ms) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                best_score = -1.0
                print("Best score reset.")

    finally:
        close_litocam(hcam)
        cv2.destroyAllWindows()
        print("Camera closed.")


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live focus feedback using the Litocam CMOS sensor."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Litocam SDK camera number. Usually 0. This is not the PC webcam index.",
    )

    parser.add_argument(
        "--mag",
        type=str,
        default="20x",
        choices=["5x", "10x", "20x", "50x", "100x"],
        help="Microscope magnification.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target display/update FPS. Litocam pull speed may limit actual FPS.",
    )

    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Disable ROI selection and use whole-frame focus score.",
    )

    args = parser.parse_args()

    run_live_focus_litocam(
        camera_number=args.camera,
        magnification=args.mag,
        target_fps=args.fps,
        use_roi=not args.no_roi,
    )