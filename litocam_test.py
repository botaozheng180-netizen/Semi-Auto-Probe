import os
import time
import ctypes
import threading
from pathlib import Path
from ctypes import (
    Structure,
    POINTER,
    byref,
    c_void_p,
    c_wchar,
    c_wchar_p,
    c_uint,
    c_int,
    c_long,
    c_ushort,
)
import numpy as np
import cv2
from datetime import datetime


# =========================
# User settings
# =========================

DLL_DIR = Path(r"C:\Program Files\Lito\LitoDigital\x64")
DLL_PATH = DLL_DIR / "litocam.dll"

MAX_CAMERAS = 16

BITS = 24  # ask SDK for 24-bit RGB/BGR output

EVENT_IMAGE = 4
E_PENDING = 0x8000000A
S_OK = 0x00000000

SCRIPT_DIR = Path(__file__).resolve().parent
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = SCRIPT_DIR / f"litocam_test_frame_{TIMESTAMP}.png"

# If the saved/displayed image has wrong orientation/color, try changing these:
FLIP_VERTICAL = False
SWAP_RGB_BGR = False

# Camera configuration settings. Adjust as needed for your scene and lighting.
def configure_camera(hcam):
    """
    Set camera acquisition settings explicitly.
    This avoids relying on LitoDigital to initialize exposure/gain first.
    """

    # Moderate exposure time. Increase if too dark, decrease if overexposed.
    # Moderate analog gain. Increase if too dark, decrease if too noisy.
    exposure_us = 5000
    auto_exposure = False
    gain = 300

    if auto_exposure:
        check_hr(
            cam.Litocam_put_AutoExpoEnable(hcam, 1), 
            "Litocam_put_AutoExpoEnable"
        )
        time.sleep(1.0)  # give auto exposure time to adjust
    else:
        check_hr(
            cam.Litocam_put_AutoExpoEnable(hcam, 0), 
            "Litocam_put_AutoExpoEnable"
        )
        check_hr(
            cam.Litocam_put_ExpoTime(hcam, exposure_us), 
            "Litocam_put_ExpoTime"
        )
        check_hr(
            cam.Litocam_put_ExpoAGain(hcam, gain), 
            "Litocam_put_ExpoAGain"
        )
        time.sleep(0.5)  # give camera time to apply settings

    actual_exposure = c_uint()
    actual_gain = c_ushort()
    auto_expo = c_int()

    check_hr(
        cam.Litocam_get_AutoExpoEnable(hcam, byref(auto_expo)),
        "Litocam_get_AutoExpoEnable"
    )
    check_hr(
        cam.Litocam_get_ExpoTime(hcam, byref(actual_exposure)),
        "Litocam_get_ExpoTime"
    )
    check_hr(
        cam.Litocam_get_ExpoAGain(hcam, byref(actual_gain)),
        "Litocam_get_ExpoAGain"
    )

    print("Auto exposure:", auto_expo.value)
    print("Exposure time / us:", actual_exposure.value)
    print("Analog gain:", actual_gain.value)


# =========================
# DLL loading
# =========================

os.add_dll_directory(str(DLL_DIR))
cam = ctypes.WinDLL(str(DLL_PATH))


# =========================
# Data structures
# =========================

class LitocamDeviceV2(Structure):
    _fields_ = [
        ("displayname", c_wchar * 64),
        ("id", c_wchar * 64),
        ("model", c_void_p),
    ]

class LitocamFrameInfoV2(Structure):
    _fields_ = [
        ("width", c_uint),
        ("height", c_uint),
        ("flag", c_uint),
        ("seq", c_uint),
        ("reserved", c_uint * 16),  # extra space for future fields
    ]


# =========================
# Function signatures
# =========================

cam.Litocam_EnumV2.argtypes = [POINTER(LitocamDeviceV2)]
cam.Litocam_EnumV2.restype = c_uint

cam.Litocam_Open.argtypes = [c_wchar_p]
cam.Litocam_Open.restype = c_void_p

cam.Litocam_Close.argtypes = [c_void_p]
cam.Litocam_Close.restype = None

cam.Litocam_get_Size.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
cam.Litocam_get_Size.restype = c_long

cam.Litocam_Stop.argtypes = [c_void_p]
cam.Litocam_Stop.restype = c_long

cam.Litocam_StartPullModeWithCallback.restype = c_long

# Use the row-pitch version of PullImageV2:
# hcam, buffer, bits, row_pitch, frame_info
cam.Litocam_PullImageWithRowPitchV2.argtypes = [
    c_void_p,
    c_void_p,
    c_int,
    c_int,
    POINTER(LitocamFrameInfoV2),
]
cam.Litocam_PullImageWithRowPitchV2.restype = c_long

# Auto exposure control functions in case needed for testing
cam.Litocam_put_AutoExpoEnable.argtypes = [c_void_p, c_int]
cam.Litocam_put_AutoExpoEnable.restype = c_long

cam.Litocam_get_AutoExpoEnable.argtypes = [c_void_p, POINTER(c_int)]
cam.Litocam_get_AutoExpoEnable.restype = c_long

cam.Litocam_put_ExpoTime.argtypes = [c_void_p, c_uint]
cam.Litocam_put_ExpoTime.restype = c_long

cam.Litocam_get_ExpoTime.argtypes = [c_void_p, POINTER(c_uint)]
cam.Litocam_get_ExpoTime.restype = c_long

cam.Litocam_put_ExpoAGain.argtypes = [c_void_p, c_ushort]
cam.Litocam_put_ExpoAGain.restype = c_long

cam.Litocam_get_ExpoAGain.argtypes = [c_void_p, POINTER(c_ushort)]
cam.Litocam_get_ExpoAGain.restype = c_long

# =========================
# Helper functions
# =========================

def hr32(hr: int) -> int:
    """Convert HRESULT to unsigned 32-bit form."""
    return hr & 0xFFFFFFFF

def check_hr(hr: int, name: str):
    """Raise error if HRESULT is not S_OK."""
    if hr32(hr) != S_OK:
        raise RuntimeError(f"{name} failed, HRESULT = {hex(hr32(hr))}")

def calc_row_pitch(width: int, bits: int) -> int:
    """
    Row pitch in bytes, aligned to 4 bytes.
    For 24-bit image, this is usually width * 3 rounded to multiple of 4.
    """
    return ((width * bits + 31) // 32) * 4

def sharpness_score(frame):
    # Simple sharpness metric using variance of Laplacian
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# =========================
# Callback handling
# =========================

image_ready = threading.Event()

CALLBACK_TYPE = ctypes.WINFUNCTYPE(None, c_uint, c_void_p)


@CALLBACK_TYPE
def event_callback(nEvent, pCallbackCtx):
    print("Callback event:", nEvent)

    if nEvent == EVENT_IMAGE:
        image_ready.set()


cam.Litocam_StartPullModeWithCallback.argtypes = [
    c_void_p,
    CALLBACK_TYPE,
    c_void_p,
]


def pull_one_frame(hcam, width: int, height: int, bits: int, timeout: float = 10.0):

    # Wait for image events and repeatedly try PullImageWithRowPitchV2.
    row_pitch = calc_row_pitch(width, bits)
    buffer_size = row_pitch * height
    buffer = (ctypes.c_ubyte * buffer_size)()

    print("Row pitch:", row_pitch)
    print("Buffer size:", buffer_size)
    print("Waiting for image event...")

    deadline = time.time() + timeout

    while time.time() < deadline:
        # Wait for callback event 4
        image_ready.wait(timeout=0.5)
        image_ready.clear()

        # Try several pulls after an image event, because first attempt may return E_PENDING
        for attempt in range(10):
            info = LitocamFrameInfoV2()

            hr = cam.Litocam_PullImageWithRowPitchV2(
                hcam,
                buffer,
                bits,
                row_pitch,
                byref(info),
            )

            code = hr32(hr)

            if code == S_OK:
                print("Pulled image successfully.")
                print("Frame width:", info.width)
                print("Frame height:", info.height)
                print("Frame sequence:", info.seq)

                return buffer, info.width, info.height, row_pitch

            if code == E_PENDING:
                print(f"Pull attempt {attempt + 1}: E_PENDING, retrying...")
                time.sleep(0.05)
                continue

            raise RuntimeError(f"Litocam_PullImageWithRowPitchV2 failed, HRESULT = {hex(code)}")

    raise RuntimeError("Timed out waiting for a usable frame.")


def buffer_to_frame(buffer, width: int, height: int, row_pitch: int):
    """Convert raw SDK buffer to OpenCV-compatible NumPy image."""
    arr = np.frombuffer(buffer, dtype=np.uint8)

    # First reshape by full row pitch, including any padding bytes.
    arr = arr.reshape((height, row_pitch))

    # Remove padding. For 24-bit image, actual pixels are width * 3 bytes per row.
    arr = arr[:, : width * 3]

    frame = arr.reshape((height, width, 3)).copy()  # make contiguous

    if FLIP_VERTICAL:
        frame = cv2.flip(frame, 0)

    if SWAP_RGB_BGR:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


# =========================
# Main program
# =========================

def main():
    devices = (LitocamDeviceV2 * MAX_CAMERAS)()
    count = cam.Litocam_EnumV2(devices)

    print("Number of cameras found:", count)

    if count == 0:
        raise RuntimeError("No Litocam camera found.")

    for i in range(count):
        print(f"\nCamera {i}")
        print("Display name:", devices[i].displayname)
        print("ID:", devices[i].id)
        print("Model pointer:", devices[i].model)

    print("\nOpening:", devices[0].displayname)

    hcam = cam.Litocam_Open(devices[0].id)

    if not hcam:
        raise RuntimeError("Litocam_Open failed. Make sure LitoDigital is closed.")

    print("Camera opened. Handle:", hcam)

    try:
        width = c_int()
        height = c_int()

        hr = cam.Litocam_get_Size(hcam, byref(width), byref(height))
        check_hr(hr, "Litocam_get_Size")

        w = width.value
        h = height.value

        print("Image size:", w, "x", h)

        configure_camera(hcam)

        hr = cam.Litocam_StartPullModeWithCallback(
            hcam,
            event_callback,
            None,
        )
        check_hr(hr, "Litocam_StartPullModeWithCallback")

        # Warm-up frames
        for i in range(5):
            print(f"Discarding warm-up frame {i + 1}")
            pull_one_frame(
                hcam,
                width=w,
                height=h,
                bits=BITS,
                timeout=10.0,
            )

        buffer, w, h, row_pitch = pull_one_frame(
            hcam,
            width=w,
            height=h,
            bits=BITS,
            timeout=10.0,
        )

        print("Returned from pull_one_frame.", flush=True)

        frame = buffer_to_frame(buffer, w, h, row_pitch)

        print("Returned from buffer_to_frame.", flush=True)
        print("About to save image...")
        print("Save path:", SAVE_PATH)
        print("Frame shape and dtype:", frame.shape, frame.dtype)
        print("Frame max, min, mean:", frame.max(), frame.min(), frame.mean())
        print("Channel means:", frame.reshape(-1, 3).mean(axis=0))
        print("Sharpness score:", sharpness_score(frame))

        ok = cv2.imwrite(str(SAVE_PATH), frame)
        print("Attempted to save image. Success:", ok)
        if ok:
            print(f"Saved full-resolution frame as: {SAVE_PATH}")
        else:
            print(f"Failed to save image to: {SAVE_PATH}")
            print("Frame shape and dtype:", frame.shape, frame.dtype)

        # Display a smaller version so it fits on screen
        max_display_width = 1280
        if w > max_display_width:
            scale = max_display_width / w
            display = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            display = frame

        cv2.imshow("Litocam frame", display)
        print("Press any key on the image window to exit.")
        cv2.waitKey(0)

    finally:
        try:
            cam.Litocam_Stop(hcam)
        except Exception:
            pass

        cam.Litocam_Close(hcam)
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()