import cv2
import numpy as np
from typing import Tuple, Optional, Dict

# Image processing pipeline for Bijels Edges Estimation using OpenCV
# Each step is documented and configurable via parameters.

EDGE_ALGOS = (
    "Sobel",
    "Canny",
    "Laplacian",
    "Prewitt",
    "Roberts",
    "Scharr",
)

class ProcessingParams:
    def __init__(
        self,
        use_hist_eq: bool = True,
        use_gaussian: bool = True,
        gaussian_kernel: Tuple[int, int] = (6, 6),
        sigma_x: float = 0.0,
        sigma_y: float = 0.0,
        edge_algo: str = "Sobel",
        edge_kernel_size: int = 3,
        canny_low: int = 50,
        canny_high: int = 150,
        quant_threshold: int = 128,
        pixel_resolution_nm: float = 1.0,
        overlay_use_zero: bool = False,
        overlay_color: Tuple[int, int, int] = (0, 0, 255),
    ):
        self.use_hist_eq = use_hist_eq
        self.use_gaussian = use_gaussian
        self.gaussian_kernel = gaussian_kernel
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.edge_algo = edge_algo if edge_algo in EDGE_ALGOS else "Sobel"
        self.edge_kernel_size = int(edge_kernel_size)
        self.canny_low = int(np.clip(canny_low, 0, 255))
        self.canny_high = int(np.clip(canny_high, 0, 255))
        self.quant_threshold = int(np.clip(quant_threshold, 0, 255))
        self.pixel_resolution_nm = float(pixel_resolution_nm)
        self.overlay_use_zero = overlay_use_zero
        self.overlay_color = overlay_color

# --- Steps ---

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


def equalize_histogram_bgr(bgr: np.ndarray, enabled: bool) -> np.ndarray:
    """
    Equalize luminance while preserving color.
    Converts BGR -> YCrCb, equalizes Y channel, converts back.
    """
    if not enabled:
        return bgr
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def gaussian_smooth(img: np.ndarray, enabled: bool, kernel: Tuple[int, int], sigma_x: float, sigma_y: float) -> np.ndarray:
    """
    Apply Gaussian denoising/smoothing to grayscale or color image.
    """
    if not enabled:
        return img
    kx, ky = kernel
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    return cv2.GaussianBlur(img, (kx, ky), sigmaX=sigma_x, sigmaY=sigma_y)


def edges(gray: np.ndarray, algo: str, params: ProcessingParams) -> np.ndarray:
    """
    Apply selected edge detection algorithm.
    Step: Apply Selected Edge Algorithm.
    """
    algo = algo if algo in EDGE_ALGOS else "Sobel"
    if algo == "Sobel":
        k = params.edge_kernel_size
        k = 3 if k not in (1,3,5,7) else k
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=k)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=k)
        absx = cv2.convertScaleAbs(gx)
        absy = cv2.convertScaleAbs(gy)
        return cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    elif algo == "Canny":
        return cv2.Canny(gray, params.canny_low, params.canny_high)
    elif algo == "Laplacian":
        k = params.edge_kernel_size
        k = 3 if k not in (1,3,5,7) else k
        lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=k)
        return cv2.convertScaleAbs(lap)
    elif algo == "Prewitt":
        kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
        ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
        gx = cv2.filter2D(gray, -1, kx)
        gy = cv2.filter2D(gray, -1, ky)
        return cv2.addWeighted(cv2.convertScaleAbs(gx),0.5, cv2.convertScaleAbs(gy),0.5, 0)
    elif algo == "Roberts":
        kx = np.array([[1,0],[0,-1]], dtype=np.float32)
        ky = np.array([[0,1],[-1,0]], dtype=np.float32)
        gx = cv2.filter2D(gray, -1, kx)
        gy = cv2.filter2D(gray, -1, ky)
        return cv2.addWeighted(cv2.convertScaleAbs(gx),0.5, cv2.convertScaleAbs(gy),0.5, 0)
    elif algo == "Scharr":
        gx = cv2.Scharr(gray, cv2.CV_16S, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_16S, 0, 1)
        absx = cv2.convertScaleAbs(gx)
        absy = cv2.convertScaleAbs(gy)
        return cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    else:
        return gray


def quantize(gray: np.ndarray, threshold: int) -> np.ndarray:
    """
    Quantize each pixel: 0 if < threshold, 255 if >= threshold.
    Step: Quantize pixels based on threshold.
    """
    thr = int(np.clip(threshold, 0, 255))
    _, binary = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return binary


def overlay_mask_on_color(color_img: np.ndarray, mask: np.ndarray, overlay_color: Tuple[int,int,int], use_zero: bool) -> np.ndarray:
    """
    Overlay each pixel value 255 over input image (color).
    Step: Overlay 255 pixels.
    """
    overlay = color_img.copy()
    target = (mask == 0) if use_zero else (mask == 255)
    overlay[target] = overlay_color
    return overlay


def process_image(bgr: np.ndarray, params: ProcessingParams):
    """
    Run the full pipeline and return intermediates + stats.
    """
    input_color = bgr
    # Keep color until just before edge detection
    eq_color = equalize_histogram_bgr(input_color, params.use_hist_eq)
    smooth_color = gaussian_smooth(eq_color, params.use_gaussian, params.gaussian_kernel, params.sigma_x, params.sigma_y)
    # Convert to gray scale just before edge detection
    smooth_gray = to_gray(smooth_color)
    # Apply Selected Edge Algorithm on grayscale
    edge_gray = edges(smooth_gray, params.edge_algo, params)
    edge_for_quant = edge_gray

    # Quantize
    quant_mask = quantize(edge_for_quant, params.quant_threshold)

    # Overlay 255 pixels on original color
    final_overlay = overlay_mask_on_color(input_color, quant_mask, params.overlay_color, params.overlay_use_zero)

    # Histograms (computed on grayscale views)
    hist_input_gray = compute_histogram(to_gray(input_color))
    hist_eq_gray = compute_histogram(to_gray(eq_color))

    # Stats
    total_pixels = quant_mask.size
    count_255 = int(np.count_nonzero(quant_mask == 255))
    percent_255 = (count_255 / total_pixels * 100.0) if total_pixels > 0 else 0.0
    count_0 = total_pixels - count_255
    percent_0 = (count_0 / total_pixels * 100.0) if total_pixels > 0 else 0.0

    stats: Dict[str, float] = {
        "hist_input_gray": hist_input_gray,
        "hist_eq_gray": hist_eq_gray,
        "count_255": count_255,
        "percent_255": percent_255,
        "count_0": count_0,
        "percent_0": percent_0,
        "total_pixels": total_pixels,
    }

    return input_color, eq_color, smooth_color, edge_gray, quant_mask, final_overlay, stats


def load_image(path: str) -> Optional[np.ndarray]:
    """Step: Open image"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def save_image(path: str, img: np.ndarray) -> bool:
    try:
        return bool(cv2.imwrite(path, img))
    except Exception:
        return False
