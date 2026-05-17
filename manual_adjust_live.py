"""
manual_adjust_live.py

Semi-automatic live focus feedback compatible with both:
- Litocam/LitoDigital cameras accessed through litocam.dll
- normal Windows/OpenCV cameras accessed through cv2.VideoCapture

Features:
- Chooses camera backend with --backend litocam/opencv/auto
- Captures frames from either backend through one common camera-session interface
- Automatically selects several candidate ROIs across the field of view
- Computes focus score as the median score over the selected ROIs
- Shows live video with selected ROI boxes drawn on screen
- Draws a small rolling focus-score chart inside the display window
- Turns the rolling focus line from red to green when repeated good frames are reached
- Press "a" to auto-reselect ROIs during focusing, "r" to reset the best score, "q" to quit

Notes:
- Use --backend litocam for cameras seen by LitoDigital.
- Use --backend opencv for cameras seen by the Windows Camera app / OpenCV.
- Use --backend auto to try Litocam first, then fall back to OpenCV.
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
# User-adjustable settings
# ============================================================

FOCUS_TARGETS_BY_BACKEND = {
    "litocam": {
        "5x": 875.0,
        "10x": 480.0,
        "20x": 210.0,
        "50x": 75.0,
        "100x": 60.0,
    },
    "opencv": {
        "5x": 415.0,
        "10x": 225.0,
        "20x": 100.0,
        "50x": 35.0,
        "100x": 25.0,
    },
}


def get_focus_target(
    magnification: str,
    backend_name: str,
    override: float | None = None,
) -> float:
    """
    Select focus target based on the actual camera backend.

    The same microscope magnification can give different Laplacian scores
    depending on camera, lens, resolution, exposure, and backend.
    """
    if override is not None:
        return float(override)

    backend_name = backend_name.lower()

    if backend_name not in FOCUS_TARGETS_BY_BACKEND:
        raise ValueError(
            f"Unsupported backend for focus targets: {backend_name}. "
            f"Choose from {list(FOCUS_TARGETS_BY_BACKEND.keys())}."
        )

    targets = FOCUS_TARGETS_BY_BACKEND[backend_name]

    if magnification not in targets:
        raise ValueError(
            f"Unsupported magnification {magnification} for backend {backend_name}. "
            f"Choose from {list(targets.keys())}."
        )

    return targets[magnification]


IMAGE_DISPLAY_WIDTH = 850
PANEL_WIDTH = 430
WINDOW_NAME = "Semi-auto Live Focus Feedback"
ROI_SELECT_WINDOW = "Select ROI -> Press Enter"

WARMUP_FRAMES = 3
CAPTURE_TIMEOUT = 10.0

CONSECUTIVE_GOOD_REQUIRED = 10
BAD_FRAMES_TO_RESET = 2

MIN_ROI_MEAN_BRIGHTNESS = 10.0
MAX_ROI_MEAN_BRIGHTNESS = 240.0
EDGE_MAX_SATURATION_FRACTION = 0.25
FRAME_MAX_SATURATION_FRACTION = 0.30

CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
EDGE_DILATE_PIXELS = 4
MIN_EDGE_PIXELS = 50

PROMPT_REL_EPSILON = 0.005
PROMPT_ABS_EPSILON = 0.5

# Automatic multi-ROI settings.
# The baseline frame may be blurry, so ROI ranking uses contrast + edge content,
# not only Canny edges.
AUTO_ROI_COUNT = 8
AUTO_ROI_GRID_ROWS = 5
AUTO_ROI_GRID_COLS = 7
AUTO_ROI_BOX_WIDTH_FRACTION = 0.16
AUTO_ROI_BOX_HEIGHT_FRACTION = 0.16
AUTO_ROI_MARGIN_FRACTION = 0.08
AUTO_ROI_MAX_OVERLAP_IOU = 0.20
AUTO_ROI_MIN_CONTRAST = 4.0
AUTO_ROI_MAX_SATURATION = 0.35
MAX_ROI_IOU = 0.15

ROI = tuple[int, int, int, int]



# ============================================================
# Camera backend abstraction
# ============================================================

def load_litocam_backend():
    """
    Load the Litocam SDK wrapper only when it is actually needed.

    This keeps the program usable with ordinary OpenCV/Windows cameras even when
    litocam.dll or the LitoDigital installation is unavailable.
    """
    candidate_module_names = [
        "litocam_test",
        "camera_test",
    ]

    last_error: Exception | None = None

    for module_name in candidate_module_names:
        try:
            module = __import__(module_name)
            return module
        except Exception as exc:
            last_error = exc

    script_dir = Path(__file__).resolve().parent
    candidate_files = [
        script_dir / "camera_test.py",
        script_dir / "litocam_test.py",
        script_dir / "camera_test(5).py",
        script_dir / "litocam_test(5).py",
    ]

    for path in candidate_files:
        if not path.exists():
            continue

        try:
            spec = importlib.util.spec_from_file_location("litocam_backend", path)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as exc:
            last_error = exc

    raise ImportError(
        "Could not load a Litocam backend. Put camera_test.py or litocam_test.py "
        "in the same folder, check the LitoDigital installation, or run with "
        "--backend opencv for Windows/OpenCV cameras."
    ) from last_error


def opencv_api_preference(api_name: str) -> int:
    """
    Convert a readable API name into an OpenCV VideoCapture backend constant.
    """
    api_name = api_name.lower()

    if api_name in {"any", "default"}:
        return cv2.CAP_ANY

    if api_name == "dshow":
        return cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY

    if api_name == "msmf":
        return cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else cv2.CAP_ANY

    raise ValueError("opencv_api must be one of: any, dshow, msmf")


class BaseCameraSession:
    """
    Small common interface used by the live focus loop.

    Each camera backend only needs:
    - open()
    - capture_frame()
    - close()
    - name / width / height fields
    """

    backend_name = "base"

    def __init__(self):
        self.name = "unopened camera"
        self.width: int | None = None
        self.height: int | None = None

    def open(self):
        raise NotImplementedError

    def capture_frame(self) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class LitocamCameraSession(BaseCameraSession):
    backend_name = "litocam"

    def __init__(
        self,
        camera_number: int = 0,
        configure_camera: bool = True,
        timeout: float = CAPTURE_TIMEOUT,
    ):
        super().__init__()
        self.camera_number = camera_number
        self.configure_camera = configure_camera
        self.timeout = timeout

        self.ct = None
        self.hcam = None
        self._callback = None

    def _event_callback(self, nEvent, pCallbackCtx):
        if self.ct is not None and nEvent == self.ct.EVENT_IMAGE:
            self.ct.image_ready.set()

    def open(self):
        self.ct = load_litocam_backend()

        devices = (self.ct.LitocamDeviceV2 * self.ct.MAX_CAMERAS)()
        count = self.ct.cam.Litocam_EnumV2(devices)

        print("Number of Litocam cameras found:", count)

        if count == 0:
            raise RuntimeError("No Litocam camera found. Close LitoDigital and check the USB connection.")

        for i in range(count):
            print(f"Litocam {i}: {devices[i].displayname}, ID = {devices[i].id}")

        if self.camera_number < 0 or self.camera_number >= count:
            raise ValueError(
                f"Invalid Litocam camera number {self.camera_number}. "
                f"Available: 0 to {count - 1}"
            )

        self.hcam = self.ct.cam.Litocam_Open(devices[self.camera_number].id)

        if not self.hcam:
            raise RuntimeError("Litocam_Open failed. Make sure LitoDigital is closed.")

        width = c_int()
        height = c_int()

        hr = self.ct.cam.Litocam_get_Size(self.hcam, byref(width), byref(height))
        self.ct.check_hr(hr, "Litocam_get_Size")

        self.width = width.value
        self.height = height.value
        self.name = f"Litocam {self.camera_number}: {devices[self.camera_number].displayname}"

        print("Opened:", self.name)
        print("Image size:", self.width, "x", self.height)

        if self.configure_camera:
            self.ct.configure_camera(self.hcam)

        self.ct.image_ready.clear()

        self._callback = self.ct.CALLBACK_TYPE(self._event_callback)

        hr = self.ct.cam.Litocam_StartPullModeWithCallback(
            self.hcam,
            self._callback,
            None,
        )
        self.ct.check_hr(hr, "Litocam_StartPullModeWithCallback")

        return self

    def capture_frame(self) -> np.ndarray:
        if self.ct is None or self.hcam is None:
            raise RuntimeError("Litocam camera is not open.")

        bits = self.ct.BITS
        row_pitch = self.ct.calc_row_pitch(self.width, bits)
        buffer_size = row_pitch * self.height
        buffer = (ctypes.c_ubyte * buffer_size)()

        deadline = time.time() + self.timeout

        while time.time() < deadline:
            self.ct.image_ready.wait(timeout=0.5)
            self.ct.image_ready.clear()

            for _attempt in range(10):
                info = self.ct.LitocamFrameInfoV2()

                hr = self.ct.cam.Litocam_PullImageWithRowPitchV2(
                    self.hcam,
                    buffer,
                    bits,
                    row_pitch,
                    byref(info),
                )

                code = self.ct.hr32(hr)

                if code == self.ct.S_OK:
                    return self.ct.buffer_to_frame(buffer, info.width, info.height, row_pitch)

                if code == self.ct.E_PENDING:
                    time.sleep(0.01)
                    continue

                raise RuntimeError(
                    f"Litocam_PullImageWithRowPitchV2 failed, HRESULT = {hex(code)}"
                )

        raise RuntimeError("Timed out waiting for a usable Litocam frame.")

    def close(self):
        if self.ct is None or self.hcam is None:
            return

        try:
            self.ct.cam.Litocam_Stop(self.hcam)
        except Exception:
            pass

        try:
            self.ct.cam.Litocam_Close(self.hcam)
        except Exception:
            pass

        self.hcam = None


class OpenCVCameraSession(BaseCameraSession):
    backend_name = "opencv"

    def __init__(
        self,
        camera_number: int = 0,
        opencv_api: str = "dshow",
        frame_width: int | None = None,
        frame_height: int | None = None,
        warmup_reads: int = 5,
    ):
        super().__init__()
        self.camera_number = camera_number
        self.opencv_api = opencv_api
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.warmup_reads = warmup_reads
        self.cap = None

    def open(self):
        api = opencv_api_preference(self.opencv_api)

        self.cap = cv2.VideoCapture(self.camera_number, api)

        if not self.cap.isOpened():
            # Fallback to CAP_ANY in case the requested Windows backend fails.
            if api != cv2.CAP_ANY:
                self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_number, cv2.CAP_ANY)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open OpenCV camera index {self.camera_number}. "
                "Try another index, e.g. --backend opencv --camera 1."
            )

        if self.frame_width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        if self.frame_height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Warm up the camera buffer/exposure.
        last_frame = None
        for _ in range(max(0, self.warmup_reads)):
            ret, frame = self.cap.read()
            if ret:
                last_frame = frame

        if last_frame is None:
            ret, last_frame = self.cap.read()
            if not ret:
                raise RuntimeError("OpenCV camera opened but did not return a frame.")

        self.height, self.width = last_frame.shape[:2]
        self.name = f"OpenCV camera {self.camera_number} ({self.opencv_api})"

        print("Opened:", self.name)
        print("Image size:", self.width, "x", self.height)

        return self

    def capture_frame(self) -> np.ndarray:
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("OpenCV camera is not open.")

        ret, frame = self.cap.read()

        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from OpenCV camera.")

        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_camera_session(
    backend: str,
    camera_number: int,
    opencv_api: str = "dshow",
    frame_width: int | None = None,
    frame_height: int | None = None,
    litocam_configure: bool = True,
) -> BaseCameraSession:
    """
    Open either a Litocam SDK camera or an OpenCV/Windows camera.

    backend:
    - "litocam": force LitoDigital/litocam.dll camera
    - "opencv": force cv2.VideoCapture camera
    - "auto": try Litocam first, then fall back to OpenCV
    """
    backend = backend.lower()

    if backend == "litocam":
        return LitocamCameraSession(
            camera_number=camera_number,
            configure_camera=litocam_configure,
        ).open()

    if backend == "opencv":
        return OpenCVCameraSession(
            camera_number = camera_number,
            opencv_api = opencv_api,
            frame_width = frame_width,
            frame_height = frame_height,
        ).open()

    if backend == "auto":
        try:
            print("Trying Litocam backend first...")
            return LitocamCameraSession(
                camera_number = camera_number,
                configure_camera = litocam_configure,
            ).open()
        except Exception as litocam_error:
            print("Litocam backend unavailable or failed:")
            print(f"  {litocam_error}")
            print("Falling back to OpenCV/Windows camera backend...")

            return OpenCVCameraSession(
                camera_number = camera_number,
                opencv_api = opencv_api,
                frame_width = frame_width,
                frame_height = frame_height,
            ).open()

    raise ValueError("backend must be one of: auto, litocam, opencv")


def warmup_camera(camera: BaseCameraSession, warmup_frames: int = WARMUP_FRAMES):
    print("\nWarming up camera...")
    for i in range(warmup_frames):
        _ = camera.capture_frame()
        print(f"Discarded warm-up frame {i + 1}/{warmup_frames}")

# ============================================================
# ROI and image analysis helpers
# ============================================================

def valid_roi(roi) -> bool:
    """
    Return True only for a single ROI tuple/list: (x, y, w, h).

    Important: in auto-ROI mode, `rois` is a list of many ROI tuples.
    We must not try to unpack that list as if it were one ROI.
    """
    if roi is None:
        return False

    if not isinstance(roi, (tuple, list)) or len(roi) != 4:
        return False

    try:
        _x, _y, w, h = roi
        return float(w) > 5 and float(h) > 5
    except (TypeError, ValueError):
        return False


def normalize_rois(rois) -> list[ROI]:
    """
    Convert None / single ROI / list of ROIs into a clean list.

    Accepted inputs:
    - None
    - one ROI: (x, y, w, h)
    - many ROIs: [(x, y, w, h), ...]
    """
    if rois is None:
        return []

    # Case 1: a single ROI tuple/list, e.g. (x, y, w, h)
    if valid_roi(rois):
        x, y, w, h = rois
        return [(int(x), int(y), int(w), int(h))]

    # Case 2: a list of ROIs, e.g. [(x, y, w, h), ...]
    if not isinstance(rois, (tuple, list)):
        return []

    clean = []
    for roi in rois:
        if valid_roi(roi):
            x, y, w, h = roi
            clean.append((int(x), int(y), int(w), int(h)))

    return clean


def resize_for_display(frame: np.ndarray, width: int = IMAGE_DISPLAY_WIDTH) -> np.ndarray:
    """
    Resize the live camera frame to a fixed display width.

    This affects only the image area. The side dashboard panel is added later
    in draw_dashboard().
    """
    h, w = frame.shape[:2]

    if w <= 0 or h <= 0:
        return frame.copy()

    scale = width / w
    new_size = (int(w * scale), int(h * scale))

    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


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
    Manual fallback: select ROI on a resized preview, then map it back to
    full-resolution coordinates.
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


def generate_candidate_rois(
    frame: np.ndarray,
    grid_rows: int = AUTO_ROI_GRID_ROWS,
    grid_cols: int = AUTO_ROI_GRID_COLS,
    box_width_fraction: float = AUTO_ROI_BOX_WIDTH_FRACTION,
    box_height_fraction: float = AUTO_ROI_BOX_HEIGHT_FRACTION,
    margin_fraction: float = AUTO_ROI_MARGIN_FRACTION,
) -> list[ROI]:
    """
    Generate stratified grid ROI candidates.

    This is intentionally not purely random. A grid gives better field coverage,
    which is useful for periodically arranged metal pads.
    """
    frame_h, frame_w = frame.shape[:2]

    box_w = max(32, int(frame_w * box_width_fraction))
    box_h = max(32, int(frame_h * box_height_fraction))

    margin_x = int(frame_w * margin_fraction)
    margin_y = int(frame_h * margin_fraction)

    x_min = margin_x + box_w // 2
    x_max = frame_w - margin_x - box_w // 2
    y_min = margin_y + box_h // 2
    y_max = frame_h - margin_y - box_h // 2

    if x_max <= x_min or y_max <= y_min:
        return [(0, 0, frame_w, frame_h)]

    xs = np.linspace(x_min, x_max, grid_cols)
    ys = np.linspace(y_min, y_max, grid_rows)

    rois = []
    for cy in ys:
        for cx in xs:
            x = int(round(cx - box_w / 2))
            y = int(round(cy - box_h / 2))

            x = max(0, min(x, frame_w - box_w))
            y = max(0, min(y, frame_h - box_h))

            rois.append((x, y, box_w, box_h))

    return rois


def roi_iou(a: ROI, b: ROI) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def roi_quality(frame: np.ndarray, roi: ROI) -> dict:
    """
    Score whether an ROI is useful for focus tracking.

    Good candidates usually have:
    - enough local contrast
    - some edge content
    - acceptable brightness
    - not too much saturation

    Because the initial frame can be blurry, contrast is included so the selector
    does not rely only on Canny edges.
    """
    target = crop_roi(frame, roi)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p5)
    std = float(np.std(gray))
    mean = float(np.mean(gray))
    saturation = float(np.mean(target >= 250))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    edge_density = float(np.mean(edges > 0))

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_ok = MIN_ROI_MEAN_BRIGHTNESS <= mean <= MAX_ROI_MEAN_BRIGHTNESS
    saturation_ok = saturation <= AUTO_ROI_MAX_SATURATION
    contrast_ok = contrast >= AUTO_ROI_MIN_CONTRAST

    usable = brightness_ok and saturation_ok and contrast_ok

    # The weights are heuristic. Contrast helps in blurred frames; edge density
    # becomes more useful as the image sharpens.
    quality = contrast + 0.5 * std + 800.0 * edge_density + 0.02 * lap_var

    if not brightness_ok:
        quality *= 0.25
    if not saturation_ok:
        quality *= 0.35
    if not contrast_ok:
        quality *= 0.35

    return {
        "roi": roi,
        "quality": float(quality),
        "usable": usable,
        "contrast": contrast,
        "std": std,
        "mean": mean,
        "saturation": saturation,
        "edge_density": edge_density,
        "lap_var": lap_var,
    }


def auto_select_rois(
    frame: np.ndarray,
    roi_count: int = AUTO_ROI_COUNT,
    grid_rows: int = AUTO_ROI_GRID_ROWS,
    grid_cols: int = AUTO_ROI_GRID_COLS,
) -> list[ROI]:
    """
    Automatically select multiple ROIs.

    Selection logic:
    1. Generate grid candidates.
    2. Score candidates by contrast + edge content + Laplacian variance.
    3. Prefer usable candidates.
    4. Select high-quality boxes while rejecting strong overlaps.
    5. If too few non-overlapping boxes are found, relax the overlap rule.
    6. If still too few, fill with the best remaining candidates.
    """
    candidates = generate_candidate_rois(
        frame,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )

    scored_candidates = [roi_quality(frame, roi) for roi in candidates]

    # Sort all candidates from best to worst
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda item: item["quality"],
        reverse=True,
    )

    # Prefer usable candidates, but keep all candidates as fallback
    usable_candidates = [
        item for item in sorted_candidates
        if item["usable"]
    ]

    if len(usable_candidates) >= max(1, roi_count // 2):
        primary_pool = usable_candidates
    else:
        primary_pool = sorted_candidates

    selected_rois: list[ROI] = []

    def already_selected(candidate_roi: ROI) -> bool:
        return candidate_roi in selected_rois

    def add_non_overlapping_from_pool(pool, max_iou: float):
        """
        Add candidates from pool if they do not overlap too much
        with already selected ROIs.
        """
        nonlocal selected_rois

        for candidate in pool:
            candidate_roi = candidate["roi"]

            if already_selected(candidate_roi):
                continue

            overlaps_existing = any(
                roi_iou(candidate_roi, existing_roi) > max_iou
                for existing_roi in selected_rois
            )

            if overlaps_existing:
                continue

            selected_rois.append(candidate_roi)

            if len(selected_rois) >= roi_count:
                break

    # Pass 1: strict non-overlap, using preferred pool
    add_non_overlapping_from_pool(
        primary_pool,
        max_iou=AUTO_ROI_MAX_OVERLAP_IOU,
    )

    # Pass 2: strict non-overlap, using all candidates
    if len(selected_rois) < roi_count:
        add_non_overlapping_from_pool(
            sorted_candidates,
            max_iou=AUTO_ROI_MAX_OVERLAP_IOU,
        )

    # Pass 3: relaxed overlap, useful when features are dense or image is small
    if len(selected_rois) < roi_count:
        relaxed_iou = max(0.35, AUTO_ROI_MAX_OVERLAP_IOU * 2.5)

        add_non_overlapping_from_pool(
            sorted_candidates,
            max_iou=relaxed_iou,
        )

    # Pass 4: final fallback, fill with best remaining boxes even if overlapping
    if len(selected_rois) < roi_count:
        for candidate in sorted_candidates:
            candidate_roi = candidate["roi"]

            if already_selected(candidate_roi):
                continue

            selected_rois.append(candidate_roi)

            if len(selected_rois) >= roi_count:
                break

    print(f"Auto-selected {len(selected_rois)} ROI(s).")
    return selected_rois


def calculate_multi_roi_focus_score(frame: np.ndarray, rois) -> float:
    """
    Compute aggregate focus score.

    With multiple ROIs, use the median so one unusually sharp/saturated/noisy box
    does not dominate the feedback.
    """
    roi_list = normalize_rois(rois)

    if not roi_list:
        return calculate_focus_score(frame, None)

    roi_scores = [calculate_focus_score(frame, roi) for roi in roi_list]
    return float(np.median(roi_scores))


def analyze_single_roi(frame: np.ndarray, roi, focus_target: float) -> dict:
    target = crop_roi(frame, roi)

    score = calculate_focus_score(frame, roi)
    mean = float(target.mean())
    min_value = int(target.min())
    max_value = int(target.max())

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
        saturation_fraction = float(np.mean(target >= 250))
        metric_source = "Whole frame"

    focus_ok = score >= focus_target
    brightness_ok = (
        MIN_ROI_MEAN_BRIGHTNESS <= mean <= MAX_ROI_MEAN_BRIGHTNESS
        and saturation_fraction <= EDGE_MAX_SATURATION_FRACTION
    )

    return {
        "score": score,
        "mean": mean,
        "min": min_value,
        "max": max_value,
        "saturation_fraction": saturation_fraction,
        "focus_ok": focus_ok,
        "brightness_ok": brightness_ok,
        "metric_source": metric_source,
    }


def analyze_frame(frame: np.ndarray, rois, focus_target: float):
    """
    Analyze one frame using either whole-frame, single-ROI, or multi-ROI scoring.
    """
    roi_list = normalize_rois(rois)
    whole_saturation_fraction = float(np.mean(frame >= 250))

    if not roi_list:
        single = analyze_single_roi(frame, None, focus_target)
        single["whole_saturation_fraction"] = whole_saturation_fraction
        single["good_focus"] = single["focus_ok"] and single["brightness_ok"]
        return single

    roi_metrics = [analyze_single_roi(frame, roi, focus_target) for roi in roi_list]

    all_scores = np.array([m["score"] for m in roi_metrics], dtype=np.float64)
    valid_scores = np.array(
        [m["score"] for m in roi_metrics if m["brightness_ok"]],
        dtype=np.float64,
    )

    if len(valid_scores) > 0:
        aggregate_score = float(np.median(valid_scores))
    else:
        aggregate_score = float(np.median(all_scores))

    means = np.array([m["mean"] for m in roi_metrics], dtype=np.float64)
    saturations = np.array([m["saturation_fraction"] for m in roi_metrics], dtype=np.float64)

    brightness_ok_count = sum(1 for m in roi_metrics if m["brightness_ok"])
    brightness_ok = brightness_ok_count >= max(1, int(np.ceil(len(roi_metrics) / 2)))

    focus_ok = aggregate_score >= focus_target
    good_focus = focus_ok and brightness_ok

    return {
        "score": aggregate_score,
        "mean": float(np.median(means)),
        "min": int(min(m["min"] for m in roi_metrics)),
        "max": int(max(m["max"] for m in roi_metrics)),
        "saturation_fraction": float(np.median(saturations)),
        "whole_saturation_fraction": whole_saturation_fraction,
        "focus_ok": focus_ok,
        "brightness_ok": brightness_ok,
        "good_focus": good_focus,
        "metric_source": (
            f"Auto multi-ROI median: {len(roi_metrics)} boxes, "
            f"{brightness_ok_count} brightness-valid"
        ),
        "roi_scores": [float(m["score"]) for m in roi_metrics],
        "roi_brightness_ok_count": brightness_ok_count,
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

    def reset(self):
        self.good_count = 0
        self.bad_count = 0

    def update(self, good_focus: bool):
        if good_focus:
            self.good_count = min(self.good_count + 1, self.required_good)
            self.bad_count = 0
        else:
            self.bad_count = min(self.bad_count + 1, self.bad_to_reset)

            if self.bad_count >= self.bad_to_reset:
                self.good_count = 0

        is_clear = self.good_count >= self.required_good
        return self.good_count, self.bad_count, is_clear


# ============================================================
# Drawing helpers
# ============================================================

def draw_roi_rectangles(display: np.ndarray, full_frame_shape, rois):
    roi_list = normalize_rois(rois)

    if not roi_list:
        return

    full_h, full_w = full_frame_shape[:2]
    display_h, display_w = display.shape[:2]

    sx = display_w / full_w
    sy = display_h / full_h

    for idx, roi in enumerate(roi_list, start=1):
        x, y, w, h = roi

        p1 = (int(x * sx), int(y * sy))
        p2 = (int((x + w) * sx), int((y + h) * sy))

        cv2.rectangle(display, p1, p2, (0, 255, 255), 2)
        cv2.putText(
            display,
            str(idx),
            (p1[0] + 4, max(18, p1[1] + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


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
    rois,
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
    backend_name: str = "",
    camera_label: str = "",
    required_good: int = CONSECUTIVE_GOOD_REQUIRED,
):
    """
    Build a stable side-panel dashboard.

    Important display choices:
    - The live image is kept clean; only ROI rectangles are drawn on it.
    - Text, prompt, controls, and chart stay on the right-side panel.
    - The prompt area has a fixed height, so the chart no longer jumps up/down
      when the prompt changes between 2 and 3 lines.
    """
    # Left: live image
    image = resize_for_display(frame, width=IMAGE_DISPLAY_WIDTH)
    roi_list = normalize_rois(rois)
    draw_roi_rectangles(image, frame.shape, roi_list)

    image_h, _image_w = image.shape[:2]

    # Right: dashboard panel
    panel = np.full((image_h, PANEL_WIDTH, 3), 35, dtype=np.uint8)

    prompt = make_prompt(previous_ema_score, ema_score)

    whole_sat_warning = metrics["whole_saturation_fraction"] > FRAME_MAX_SATURATION_FRACTION
    status_text = "CLEAR" if is_clear else "ADJUSTING"
    status_color = (0, 180, 0) if is_clear else (0, 0, 255)
    roi_mode_text = f"{len(roi_list)} auto ROI(s)" if roi_list else "whole frame"

    # ----------------------------
    # Top status block
    # ----------------------------
    lines = [
        f"Magnification: {magnification} | Scoring: {roi_mode_text}",
        f"EMA: {ema_score:.2f} | Raw: {raw_score:.2f}",
        f"Best: {best_score:.2f} | Target: {focus_target:.2f}",
        f"Status: {status_text} | Good: {good_count}/{required_good}",
        f"Brightness OK: {metrics['brightness_ok']} | Focus OK: {metrics['focus_ok']}",
        f"ROI mean: {metrics['mean']:.1f} | ROI min/max: {metrics['min']}/{metrics['max']}",
        f"ROI sat: {metrics['saturation_fraction']:.4f} | Frame sat: {metrics['whole_saturation_fraction']:.4f}",
        f"Sat warning: {whole_sat_warning}",
    ]

    x = 15
    y = 30
    line_gap = 24

    for line in lines:
        color = status_color if line.startswith("Status") else (255, 255, 255)
        cv2.putText(
            panel,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        y += line_gap

    # ----------------------------
    # Fixed-height prompt area
    # ----------------------------
    prompt_top = y + 14
    prompt_line_gap = 22
    prompt_max_lines = 3
    prompt_max_chars = 38

    prompt_words = prompt.split()
    prompt_lines = []
    current = ""

    for word in prompt_words:
        candidate = (current + " " + word).strip()
        if len(candidate) > prompt_max_chars and current:
            prompt_lines.append(current)
            current = word
        else:
            current = candidate

    if current:
        prompt_lines.append(current)

    # Draw at most three lines, but reserve exactly three lines of vertical space
    # so the chart y-position is stable.
    for i in range(prompt_max_lines):
        line = prompt_lines[i] if i < len(prompt_lines) else ""
        cv2.putText(
            panel,
            line,
            (x, prompt_top + i * prompt_line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    # ----------------------------
    # Fixed-position chart area
    # ----------------------------
    controls_y = image_h - 54
    backend_y = image_h - 24

    chart_h = 165
    preferred_chart_y = prompt_top + prompt_max_lines * prompt_line_gap + 28
    max_chart_y = controls_y - chart_h - 28
    chart_y = min(preferred_chart_y, max_chart_y)

    # In very small windows, keep a usable chart and avoid negative y.
    chart_y = max(prompt_top + prompt_max_lines * prompt_line_gap + 10, chart_y)
    chart_y = min(chart_y, max(10, image_h - chart_h - 95))

    panel = draw_focus_chart(
        panel,
        scores=scores,
        threshold=focus_target,
        is_clear=is_clear,
        x=15,
        y=chart_y,
        w=PANEL_WIDTH - 30,
        h=chart_h,
    )

    # ----------------------------
    # Bottom controls and backend information
    # ----------------------------
    controls_text = "Keys: q quit | r reset best | a auto-ROIs"
    cv2.putText(
        panel,
        controls_text,
        (15, controls_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    bottom_text = f"Backend: {backend_name} | Camera: {camera_label}"
    cv2.putText(
        panel,
        bottom_text[:52],
        (15, backend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    dashboard = np.hstack([image, panel])
    return dashboard


# ============================================================
# Main live loop
# ============================================================

def run_live_focus(
    backend: str = "auto",
    camera_number: int = 0,
    magnification: str = "20x",
    target_fps: float = 10.0,
    roi_mode: str = "auto",
    auto_roi_count: int = AUTO_ROI_COUNT,
    opencv_api: str = "dshow",
    frame_width: int | None = None,
    frame_height: int | None = None,
    litocam_configure: bool = True,
    good_required: int = CONSECUTIVE_GOOD_REQUIRED,
    focus_target_override: float | None = None,
):
    valid_magnifications = set()
    for targets in FOCUS_TARGETS_BY_BACKEND.values():
        valid_magnifications.update(targets.keys())
    
    if magnification not in valid_magnifications:
        raise ValueError(
            f"Invalid magnification '{magnification}'." 
            f"Valid options: {sorted(valid_magnifications)}")

    if roi_mode not in {"auto", "manual", "none"}:
        raise ValueError("roi_mode must be 'auto', 'manual', or 'none'.")

    camera: BaseCameraSession | None = None

    try:
        camera = open_camera_session(
            backend=backend,
            camera_number=camera_number,
            opencv_api=opencv_api,
            frame_width=frame_width,
            frame_height=frame_height,
            litocam_configure=litocam_configure,
        )

        focus_target = get_focus_target(
            magnification=magnification,
            backend_name=camera.backend_name,
            override=focus_target_override,
        )

        warmup_camera(camera, WARMUP_FRAMES)

        print("\nCapturing baseline frame...")
        first_frame = camera.capture_frame()

        if roi_mode == "auto":
            rois = auto_select_rois(first_frame, roi_count=auto_roi_count)
        elif roi_mode == "manual":
            manual_roi = select_contact_roi(first_frame)
            rois = [manual_roi] if valid_roi(manual_roi) else []
        else:
            rois = []

        scores = deque(maxlen=120)
        tracker = ConsecutiveFocusTracker(
            required_good=good_required,
            bad_to_reset=BAD_FRAMES_TO_RESET,
        )

        ema_score = None
        previous_ema_score = None
        best_score = -1.0

        dt = 1.0 / target_fps if target_fps > 0 else 0.0

        print("\nSemi-auto focus feedback started.")
        print("Press q in the OpenCV window to quit.")
        print("Press r to reset best score.")
        print("Press a to auto-reselect ROIs.")
        print(f"Backend: {camera.backend_name}")
        print(f"Camera: {camera.name}")
        print(f"Magnification: {magnification}")
        print(f"Focus target: {focus_target:.2f}")
        print(f"Required good frames: {good_required}")
        print(f"ROI mode: {roi_mode}")
        print(f"Selected ROI count: {len(rois)}")

        while True:
            loop_start = time.perf_counter()

            frame = camera.capture_frame()

            raw_score = calculate_multi_roi_focus_score(frame, rois)
            previous_ema_score = ema_score
            ema_score = update_ema(ema_score, raw_score, alpha=0.35)

            metrics = analyze_frame(frame, rois, focus_target)

            # Use smoothed aggregate score for focus decision, but keep brightness
            # decision from per-ROI brightness/saturation checks.
            smoothed_focus_ok = ema_score >= focus_target
            good_focus = smoothed_focus_ok and metrics["brightness_ok"]

            good_count, bad_count, is_clear = tracker.update(good_focus)

            if ema_score > best_score:
                best_score = ema_score

            scores.append(ema_score)

            display = draw_dashboard(
                frame = frame,
                rois = rois,
                metrics = metrics,
                ema_score = ema_score,
                raw_score = raw_score,
                focus_target = focus_target,
                magnification = magnification,
                good_count = good_count,
                bad_count = bad_count,
                is_clear = is_clear,
                best_score = best_score,
                previous_ema_score = previous_ema_score,
                scores = scores,
                backend_name = camera.backend_name,
                camera_label = camera.name,
                required_good = good_required,
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

            if key == ord("a"):
                rois = auto_select_rois(frame, roi_count=auto_roi_count)
                scores.clear()
                tracker.reset()
                ema_score = None
                previous_ema_score = None
                best_score = -1.0
                print("Auto ROI reselection complete. Score history and stability count reset.")

    finally:
        if camera is not None:
            camera.close()

        cv2.destroyAllWindows()
        print("Camera closed.")


# Backward-compatible function name from earlier versions.
def run_live_focus_litocam(
    camera_number: int = 0,
    magnification: str = "20x",
    target_fps: float = 10.0,
    roi_mode: str = "auto",
    auto_roi_count: int = AUTO_ROI_COUNT,
    good_required: int = CONSECUTIVE_GOOD_REQUIRED,
    focus_target_override: float | None = None,
):
    run_live_focus(
        backend="litocam",
        camera_number=camera_number,
        magnification=magnification,
        target_fps=target_fps,
        roi_mode=roi_mode,
        auto_roi_count=auto_roi_count,
        good_required=good_required,
        focus_target_override=focus_target_override,
    )


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semi-auto live focus feedback using either Litocam SDK or OpenCV/Windows cameras."
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "litocam", "opencv"],
        help=(
            "Camera backend. "
            "litocam = LitoDigital/litocam.dll camera; "
            "opencv = Windows/OpenCV camera; "
            "auto = try litocam first, then opencv."
        ),
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help=(
            "Camera number. For litocam, this is the Litocam SDK camera number. "
            "For opencv, this is the cv2.VideoCapture index."
        ),
    )

    parser.add_argument(
        "--opencv-api",
        type=str,
        default="dshow",
        choices=["any", "dshow", "msmf"],
        help="OpenCV capture API to use when --backend opencv or auto fallback is used.",
    )

    parser.add_argument(
        "--focus-target",
        type=float,
        default=None,
        help="Override backend-specific focus target. Useful for calibration/testing.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional requested frame width for OpenCV cameras.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional requested frame height for OpenCV cameras.",
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
        help="Target display/update FPS. Actual FPS depends on camera and processing speed.",
    )

    parser.add_argument(
        "--roi-mode",
        type=str,
        default="auto",
        choices=["auto", "manual", "none"],
        help="ROI mode: auto = automatic multi-ROI, manual = old manual selection, none = whole frame.",
    )

    parser.add_argument(
        "--roi-count",
        type=int,
        default=AUTO_ROI_COUNT,
        help="Number of automatic ROIs to keep when --roi-mode auto is used.",
    )

    parser.add_argument(
        "--good-required",
        type=int,
        default=CONSECUTIVE_GOOD_REQUIRED,
        help="Number of consecutive good frames required before status becomes CLEAR.",
    )

    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Backward-compatible shortcut for --roi-mode none.",
    )

    parser.add_argument(
        "--skip-litocam-config",
        action="store_true",
        help="Do not call configure_camera() for Litocam. Useful if camera settings were already adjusted elsewhere.",
    )

    args = parser.parse_args()

    selected_roi_mode = "none" if args.no_roi else args.roi_mode

    run_live_focus(
        backend=args.backend,
        camera_number=args.camera,
        magnification=args.mag,
        target_fps=args.fps,
        roi_mode=selected_roi_mode,
        auto_roi_count=args.roi_count,
        opencv_api=args.opencv_api,
        frame_width=args.width,
        frame_height=args.height,
        litocam_configure=not args.skip_litocam_config,
        good_required=args.good_required,
        focus_target_override=args.focus_target,
    )