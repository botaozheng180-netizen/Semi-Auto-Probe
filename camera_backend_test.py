"""
camera_backend_test.py

Small test utility for checking which camera backend works before running
manual_adjust_live.py.

Use this when you are not sure whether the microscope camera is available through:
- LitoDigital / litocam.dll, or
- Windows Camera / OpenCV VideoCapture.

Examples:
    python camera_backend_test.py --list-opencv
    python camera_backend_test.py --backend litocam --camera 0
    python camera_backend_test.py --backend opencv --camera 0
    python camera_backend_test.py --backend opencv --camera 1 --opencv-api dshow
    python camera_backend_test.py --backend auto --camera 0
"""

from __future__ import annotations

import argparse
import importlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def load_focus_module():
    """
    Import manual_adjust_live module.

    Fallback if you keep the downloaded file name:
        manual_adjust_live_dual_backend.py
    """
    errors = []

    for module_name in ["manual_adjust_live", "manual_adjust_live_dual_backend"]:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append((module_name, exc))
            continue

        if hasattr(module, "open_camera_session"):
            return module

        errors.append((module_name, RuntimeError("module has no open_camera_session()")))

    message = ["Could not import the adapted manual_adjust_live module."]
    message.append("Make sure camera_backend_test.py is in the same folder as manual_adjust_live.py.")
    for module_name, exc in errors:
        message.append(f"- {module_name}: {exc}")

    raise ImportError("\n".join(message))


def sharpness_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def opencv_api_preference(api_name: str) -> int:
    api_name = api_name.lower()

    if api_name in {"any", "default"}:
        return cv2.CAP_ANY

    if api_name == "dshow":
        return cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY

    if api_name == "msmf":
        return cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else cv2.CAP_ANY

    raise ValueError("opencv_api must be one of: any, dshow, msmf")


def scan_opencv_cameras(max_index: int, opencv_api: str):
    """
    Try OpenCV camera indices and print which ones return frames.
    """
    api = opencv_api_preference(opencv_api)

    print(f"Scanning OpenCV cameras 0 to {max_index} using API: {opencv_api}")

    found = []

    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, api)

        if not cap.isOpened():
            print(f"[{index}] not opened")
            cap.release()
            continue

        # Some cameras need a few reads before returning a meaningful frame.
        frame = None
        for _ in range(5):
            ret, candidate = cap.read()
            if ret and candidate is not None:
                frame = candidate

        if frame is None:
            print(f"[{index}] opened, but no frame returned")
            cap.release()
            continue

        h, w = frame.shape[:2]
        mean = float(frame.mean())
        score = sharpness_score(frame)

        print(f"[{index}] OK | shape={w}x{h} | mean={mean:.1f} | sharpness={score:.1f}")
        found.append(index)

        cap.release()

    if found:
        print("OpenCV indices returning frames:", found)
    else:
        print("No OpenCV cameras returned frames.")


def display_frame(window_name: str, frame: np.ndarray, max_width: int = 1280):
    h, w = frame.shape[:2]

    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow(window_name, frame)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_backend(args):
    mal = load_focus_module()

    camera = None

    try:
        camera = mal.open_camera_session(
            backend=args.backend,
            camera_number=args.camera,
            opencv_api=args.opencv_api,
            frame_width=args.width,
            frame_height=args.height,
            litocam_configure=not args.skip_litocam_config,
        )

        mal.warmup_camera(camera, args.warmup)

        frame = camera.capture_frame()

        h, w = frame.shape[:2]
        print("\nCaptured test frame.")
        print("Backend:", camera.backend_name)
        print("Camera:", camera.name)
        print("Frame shape:", frame.shape)
        print("Width x height:", w, "x", h)
        print("dtype:", frame.dtype)
        print("min / max / mean:", int(frame.min()), int(frame.max()), float(frame.mean()))
        print("channel means:", frame.reshape(-1, frame.shape[-1]).mean(axis=0))
        print("sharpness score:", sharpness_score(frame))

        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path.cwd() / f"camera_backend_test_{camera.backend_name}_{timestamp}.png"
            ok = cv2.imwrite(str(path), frame)
            print("Saved frame:", path if ok else "save failed")

        if args.display:
            display_frame("Camera backend test", frame)

    finally:
        if camera is not None:
            camera.close()


def main():
    parser = argparse.ArgumentParser(description="Test Litocam and OpenCV camera backends.")

    parser.add_argument(
        "--list-opencv",
        action="store_true",
        help="Scan OpenCV/Windows camera indices and exit.",
    )

    parser.add_argument(
        "--max-index",
        type=int,
        default=5,
        help="Maximum OpenCV camera index to scan with --list-opencv.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "litocam", "opencv"],
        help="Camera backend to test.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera number/index. Litocam SDK index for litocam; cv2.VideoCapture index for opencv.",
    )

    parser.add_argument(
        "--opencv-api",
        type=str,
        default="dshow",
        choices=["any", "dshow", "msmf"],
        help="OpenCV capture API for Windows/OpenCV cameras.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional requested OpenCV frame width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional requested OpenCV frame height.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warm-up frames to discard.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save one captured frame to the current folder.",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the captured frame in an OpenCV window.",
    )

    parser.add_argument(
        "--skip-litocam-config",
        action="store_true",
        help="Do not call configure_camera() for Litocam cameras.",
    )

    args = parser.parse_args()

    if args.list_opencv:
        scan_opencv_cameras(args.max_index, args.opencv_api)
        return

    test_backend(args)


if __name__ == "__main__":
    main()