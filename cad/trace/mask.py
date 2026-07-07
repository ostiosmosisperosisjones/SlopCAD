"""
cad/trace/mask.py

Stage 1 of the V2 tracer: decode + grayscale + blur + threshold, producing a
boolean foreground mask.  Pure Pillow / scipy / numpy — no OpenCV.

The mask semantics deliberately match V1's cv2 path so Milestone A is a
behavior-preserving swap:

  gray       : cv2 BGR->GRAY weights (0.299 R + 0.587 G + 0.114 B, rounded)
  blur       : Gaussian; kernel radius `blur` px -> cv2-equivalent sigma
  threshold  : foreground = gray > threshold   (strictly greater, like cv2
               THRESH_BINARY); invert flips to gray <= threshold

The one place scipy and cv2 cannot be bit-identical is the Gaussian blur
(different kernel truncation), but on a *thresholded* mask that only perturbs a
thin fringe of edge pixels — see the parity test's tolerance.
"""

from __future__ import annotations
import numpy as np


# cv2's default BGR->GRAY (and RGB->GRAY) luminance weights.
_R_W, _G_W, _B_W = 0.299, 0.587, 0.114


def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to single-channel uint8 gray, matching cv2's weights.

    Accepts:
      - HxW           uint8 (already gray) -> returned as-is
      - HxWx3         RGB uint8
      - HxWx4         RGBA uint8 (alpha ignored)

    NOTE: channel order is **RGB** here (Pillow's native order), unlike V1's
    cv2 path which received BGR from cv2.imread.  The luminance weights are the
    same set; callers that decode via Pillow get correct results, and the
    grayscale value is symmetric enough that mask parity holds.
    """
    if image.ndim == 2:
        return image if image.dtype == np.uint8 else image.astype(np.uint8)
    chans = image.shape[2]
    if chans not in (3, 4):
        raise ValueError(f"unsupported channel count: {chans}")
    r = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 2].astype(np.float64)
    gray = _R_W * r + _G_W * g + _B_W * b
    # cv2 rounds to nearest, then clips to uint8.
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def load_gray(path: str) -> np.ndarray:
    """Decode an image file to uint8 gray via Pillow (replaces cv2.imread)."""
    from PIL import Image
    with Image.open(path) as im:
        # Convert to luminance with Pillow's own weights would differ slightly
        # from cv2; go through RGB + our weighted to_gray for parity instead.
        arr = np.asarray(im.convert("RGB"))
    return to_gray(arr)


def _cv2_equiv_sigma(k: int) -> float:
    """The sigma cv2.GaussianBlur derives from an odd kernel size when sigma=0.

    cv2 uses sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 for ksize>1.
    """
    return 0.3 * ((k - 1) * 0.5 - 1.0) + 0.8


def gaussian_blur(gray: np.ndarray, blur_radius: int) -> np.ndarray:
    """
    Gaussian blur matching V1's `k = blur*2+1` kernel convention.

    blur_radius <= 0 is a no-op.  Uses scipy.ndimage with the cv2-equivalent
    sigma so the blur strength tracks V1's slider values.
    """
    if not blur_radius or blur_radius <= 0:
        return gray
    from scipy.ndimage import gaussian_filter
    k = int(blur_radius) * 2 + 1
    sigma = _cv2_equiv_sigma(k)
    # gaussian_filter works in float; round back to uint8 to mirror cv2 output.
    out = gaussian_filter(gray.astype(np.float64), sigma=sigma, mode="nearest")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def threshold_mask(gray: np.ndarray, threshold: int,
                   invert: bool = False) -> np.ndarray:
    """
    Boolean foreground mask.

    Default (invert=False): foreground = gray > threshold  (matches cv2
    THRESH_BINARY, which is strictly greater).
    invert=True:            foreground = gray <= threshold  (THRESH_BINARY_INV).
    """
    if invert:
        return gray <= threshold
    return gray > threshold


def build_mask(image_or_gray: np.ndarray, threshold: int,
               blur_radius: int = 0, invert: bool = False) -> np.ndarray:
    """
    Full stage-1 pipeline: (optional gray) -> blur -> threshold -> bool mask.

    Accepts either a gray HxW array or a colour HxWx3/4 array.
    """
    gray = to_gray(image_or_gray)
    gray = gaussian_blur(gray, blur_radius)
    return threshold_mask(gray, threshold, invert)


# ---------------------------------------------------------------------------
# Pixel preprocessing — the "colour levers" applied BEFORE thresholding.
#
# This is the OMAX-style front end: tune how colour collapses to gray, remap
# tone, and denoise, all before the mask is formed, so the user can dial the
# pixels until the subject separates cleanly from the background.  The processed
# gray is what the tracer sees (and what the preview shows).
# ---------------------------------------------------------------------------

from dataclasses import dataclass


# Channel-collapse modes for RGB → gray.
CHANNEL_MODES = ("luminance", "red", "green", "blue", "max", "min", "average")


@dataclass
class PreprocParams:
    channel:     str   = "luminance"  # one of CHANNEL_MODES
    black_point: int   = 0            # levels: input value mapped to 0
    white_point: int   = 255          # levels: input value mapped to 255
    gamma:       float = 1.0          # tone curve exponent (1.0 = linear)
    blur:        int   = 0            # gaussian radius (px), reuses convention
    median:      int   = 0            # median denoise radius (px, 0 = off)


def collapse_channel(image: np.ndarray, mode: str) -> np.ndarray:
    """RGB(A)/gray → single-channel uint8 by the chosen channel mode."""
    if image.ndim == 2:
        return image if image.dtype == np.uint8 else image.astype(np.uint8)
    r = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 2].astype(np.float64)
    if mode == "red":
        out = r
    elif mode == "green":
        out = g
    elif mode == "blue":
        out = b
    elif mode == "max":
        out = np.maximum(np.maximum(r, g), b)
    elif mode == "min":
        out = np.minimum(np.minimum(r, g), b)
    elif mode == "average":
        out = (r + g + b) / 3.0
    else:  # luminance (default)
        out = _R_W * r + _G_W * g + _B_W * b
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def apply_levels(gray: np.ndarray, black_point: int, white_point: int,
                 gamma: float) -> np.ndarray:
    """Remap [black_point, white_point] → [0, 255] with a gamma curve.

    Stretches contrast so washed-out images binarize cleanly; gamma bends the
    midtones.  A no-op remap (0..255, gamma 1) returns the input unchanged."""
    bp = float(min(black_point, white_point))
    wp = float(max(black_point, white_point))
    if wp - bp < 1e-6:
        wp = bp + 1.0
    g = (gray.astype(np.float64) - bp) / (wp - bp)
    g = np.clip(g, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6 and gamma > 1e-6:
        g = np.power(g, 1.0 / gamma)
    return np.clip(np.rint(g * 255.0), 0, 255).astype(np.uint8)


def median_denoise(gray: np.ndarray, radius: int) -> np.ndarray:
    """Edge-preserving median filter (better than blur for speckle/JPEG noise)."""
    if not radius or radius <= 0:
        return gray
    from scipy.ndimage import median_filter
    size = int(radius) * 2 + 1
    return median_filter(gray, size=size, mode="nearest")


def preprocess_gray(image: np.ndarray, pp: PreprocParams) -> np.ndarray:
    """
    Apply the full colour/tone/noise front end to an RGB(A)/gray image and
    return the processed uint8 gray that the tracer will threshold.

    Order: channel-collapse → levels(+gamma) → median → gaussian blur.
    (Median before blur: remove speckle first, then soften remaining edges.)
    """
    gray = collapse_channel(image, pp.channel)
    gray = apply_levels(gray, pp.black_point, pp.white_point, pp.gamma)
    gray = median_denoise(gray, pp.median)
    gray = gaussian_blur(gray, pp.blur)
    return gray


def build_mask_pp(image: np.ndarray, pp: PreprocParams, threshold: int,
                  invert: bool = False):
    """Preprocess an image then threshold it → (mask, processed_gray).

    Returns both so the caller (preview) can display the exact gray the mask
    was formed from."""
    gray = preprocess_gray(image, pp)
    return threshold_mask(gray, threshold, invert), gray
