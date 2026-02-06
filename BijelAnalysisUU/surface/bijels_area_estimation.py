import cv2
import numpy as np
from typing import Tuple, Optional

# Image processing pipeline for Bijels Area Estimation using OpenCV
# Each step is documented and can be toggled/configured via parameters.

class ProcessingParams:
    def __init__(
        self,
        use_hist_eq: bool = True,
        use_gaussian: bool = True,
        gaussian_kernel: Tuple[int, int] = (6, 6),
        sigma_x: float = 0.0,
        sigma_y: float = 0.0,
        quant_threshold: int = 128,
        overlay_use_zero: bool = False,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
    ):
        self.use_hist_eq = use_hist_eq
        self.use_gaussian = use_gaussian
        self.gaussian_kernel = gaussian_kernel
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.quant_threshold = int(np.clip(quant_threshold, 0, 255))
        # When True, overlay pixels where mask==0; otherwise where mask==255
        self.overlay_use_zero = overlay_use_zero
        # Overlay color in BGR
        self.overlay_color = overlay_color


def compute_histogram(gray: np.ndarray) -> np.ndarray:
    """
    Compute histogram for a grayscale image.
    Returns a 256-length histogram array.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten()


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR color image to grayscale.
    Step: Convert to gray scale.
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def equalize_histogram(gray: np.ndarray, enabled: bool) -> np.ndarray:
    """
    Equalize the histogram of a grayscale image if enabled.
    Step: Equalize Histogram.
    """
    if not enabled:
        return gray
    return cv2.equalizeHist(gray)


def gaussian_smooth(gray: np.ndarray, enabled: bool, kernel: Tuple[int, int], sigma_x: float, sigma_y: float) -> np.ndarray:
    """
    Apply Gaussian denoising/smoothing if enabled.
    Step: Apply Gaussian denoising / smoothing.
    """
    if not enabled:
        return gray
    kx, ky = kernel
    # Ensure odd kernel sizes for GaussianBlur; if even, increment to next odd
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    return cv2.GaussianBlur(gray, (kx, ky), sigmaX=sigma_x, sigmaY=sigma_y)


def quantize(gray: np.ndarray, threshold: int) -> np.ndarray:
    """
    Quantize each pixel: 0 if < threshold, 255 if >= threshold.
    Step: Quantize pixels based on threshold.
    """
    thr = int(np.clip(threshold, 0, 255))
    _, binary = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return binary


def overlay_mask_on_color(
    color_img: np.ndarray,
    mask: np.ndarray,
    overlay_color: Tuple[int, int, int] = (0, 255, 0),
    use_zero: bool = True,
) -> np.ndarray:
    """
    Overlay GREEN pixels where the quantized mask has value 0; otherwise keep original pixel.
    New behavior: mark zeros (mask == 0) in green (BGR: (0, 255, 0)).
    """
    overlay = color_img.copy()
    target_pixels = (mask == 0) if use_zero else (mask == 255)
    overlay[target_pixels] = overlay_color
    return overlay


def process_image(
    bgr: np.ndarray,
    params: ProcessingParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    input_color = bgr
    gray = to_gray(input_color)

    # Step: Equalize Histogram (conditional)
    eq_gray = equalize_histogram(gray, params.use_hist_eq)

    # Step: Apply Gaussian denoising / smoothing (conditional)
    smooth_gray = gaussian_smooth(eq_gray, params.use_gaussian, params.gaussian_kernel, params.sigma_x, params.sigma_y)

    # Step: Quantize each pixel based on threshold
    quant_mask = quantize(smooth_gray, params.quant_threshold)

    # Step: Overlay mask on original color image
    final_overlay = overlay_mask_on_color(
        input_color,
        quant_mask,
        overlay_color=params.overlay_color,
        use_zero=params.overlay_use_zero,
    )

    # Histograms for displays
    hist_input_gray = compute_histogram(gray)
    hist_eq_gray = compute_histogram(eq_gray)
    hist_smooth_gray = compute_histogram(smooth_gray)
    hist_quant = compute_histogram(quant_mask)

    # Stats
    total_pixels = quant_mask.size
    count_0 = int(np.count_nonzero(quant_mask == 0))
    percent_0 = (count_0 / total_pixels * 100.0) if total_pixels > 0 else 0.0

    stats = {
        "hist_input_gray": hist_input_gray,
        "hist_eq_gray": hist_eq_gray,
        "hist_smooth_gray": hist_smooth_gray,
        "hist_quant": hist_quant,
        "count_0": count_0,
        "percent_0": percent_0,
        "total_pixels": total_pixels,
    }

    return input_color, gray, eq_gray, smooth_gray, quant_mask, final_overlay, stats


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Step: Open image
    Load an image from disk as BGR using OpenCV. Returns None if load fails.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def save_image(path: str, img: np.ndarray) -> bool:
    """
    Save a BGR image to disk. Returns True on success.
    """
    try:
        return bool(cv2.imwrite(path, img))
    except Exception:
        return False
