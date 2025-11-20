# preprocessing.py
"""
Preprocessing pipeline for Malaria RBC cropped images.

Contract:
def process_image(input_path: str, output_dir: str) -> list:
    Save intermediate images to output_dir with names: <base>_<step>.png
    Return: list of tuples [(filename, display_name), ...]

Steps implemented (tuned for the Kaggle cropped-cell images):
 1. Resize + normalize
 2. White balance (gray-world)
 3. LAB -> CLAHE on L-channel
 4. Extract A-channel (parasite enhancer)
 5. Bilateral filter (edge-preserving smoothing)
 6. Adaptive threshold (Gaussian)
 7. Morphological cleanup (remove small specks, close holes)
 8. Contour extraction -> final mask
 9. Overlay mask on original image
"""

import os
import cv2
import numpy as np
from skimage import exposure, img_as_ubyte
from typing import List, Tuple

# ------------- parameters you can tweak -------------
MAX_SIDE = 256             # resize longest side to this (keeps processing fast)
CLAHE_CLIP = 0.03          # CLAHE clip limit (low for microscopy)
BILATERAL_D = 7            # bilateral filter diameter
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
ADAPTIVE_BLOCKSIZE = 11    # must be odd
ADAPTIVE_C = 2
MIN_OBJECT_AREA = 40       # remove blobs smaller than this (in pixels) - tweak with image scale
MORPH_KERNEL = 3           # morphological kernel size
# ---------------------------------------------------

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _save_rgb(img_rgb: np.ndarray, path: str):
    """
    Save an RGB image (HWC, uint8). Convert to BGR for OpenCV imwrite.
    Accepts grayscale as 2D array too.
    """
    if img_rgb is None:
        raise ValueError("Empty image to save for path: " + path)
    if img_rgb.ndim == 2:
        cv2.imwrite(path, img_rgb)
    else:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)

def _resize_max_side(img: np.ndarray, max_side: int):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img.copy()

def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    """
    Simple gray-world white balance on RGB image.
    """
    # avoid division by zero
    img = img_rgb.astype(np.float32) + 1e-6
    avgR = np.mean(img[:, :, 0])
    avgG = np.mean(img[:, :, 1])
    avgB = np.mean(img[:, :, 2])
    avgGray = (avgR + avgG + avgB) / 3.0
    r_scale = avgGray / avgR
    g_scale = avgGray / avgG
    b_scale = avgGray / avgB
    balanced = img.copy()
    balanced[:, :, 0] = np.clip(balanced[:, :, 0] * r_scale, 0, 255)
    balanced[:, :, 1] = np.clip(balanced[:, :, 1] * g_scale, 0, 255)
    balanced[:, :, 2] = np.clip(balanced[:, :, 2] * b_scale, 0, 255)
    return balanced.astype(np.uint8)

def clahe_on_l_channel(img_rgb: np.ndarray, clip_limit: float = CLAHE_CLIP) -> np.ndarray:
    """
    Convert to LAB, apply CLAHE to L channel, return RGB image after merging.
    clip_limit: fraction between 0 and 1 (skimage expects float input for equalize_adapthist).
    We'll use skimage.exposure.equalize_adapthist for a robust result.
    """
    # convert to LAB using OpenCV (it expects BGR)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)  # L A B (OpenCV scales L:0-255)
    l, a, b = cv2.split(lab)
    # apply CLAHE using cv2
    clahe = cv2.createCLAHE(clipLimit=max(1.0, clip_limit*10), tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return rgb

def extract_a_channel(img_rgb: np.ndarray) -> np.ndarray:
    """
    Return the 'A' channel from LAB color space scaled to uint8 (0-255).
    In LAB, A channel encodes green-red; parasites (purple) often stand out here.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # a is 0..255 (OpenCV), return as-is (uint8)
    return a

def bilateral_smooth(img_gray: np.ndarray, d=BILATERAL_D, sigma_color=BILATERAL_SIGMA_COLOR, sigma_space=BILATERAL_SIGMA_SPACE) -> np.ndarray:
    return cv2.bilateralFilter(img_gray, d, sigma_color, sigma_space)

def adaptive_threshold(img_gray: np.ndarray, blocksize=ADAPTIVE_BLOCKSIZE, C=ADAPTIVE_C) -> np.ndarray:
    # ensure blocksize is odd and >1
    if blocksize % 2 == 0:
        blocksize += 1
    thr = cv2.adaptiveThreshold(img_gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,
                                blocksize, C)
    return thr

def morphological_cleanup(mask: np.ndarray, kernel_size=MORPH_KERNEL, min_area=MIN_OBJECT_AREA) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # close then open to fill holes and remove noise
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # remove small objects by contour filtering
    contours, _ = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(opened)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)
    return cleaned

def contours_to_filled_mask(mask: np.ndarray, min_area=MIN_OBJECT_AREA) -> np.ndarray:
    """
    Recompute mask by filling contours and filtering by area. Returns binary mask uint8.
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    newm = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(newm, [cnt], -1, 255, -1)
    return newm

def overlay_mask_on_rgb(img_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    """
    Overlay binary mask on RGB image. color is RGB tuple (e.g., red).
    alpha controls mask opacity.
    """
    overlay = img_rgb.copy().astype(np.float32)
    mask_bool = (mask > 0)
    # create color layer
    color_layer = np.zeros_like(overlay, dtype=np.float32)
    color_layer[..., 0] = color[0]
    color_layer[..., 1] = color[1]
    color_layer[..., 2] = color[2]
    # blend where mask is true
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color_layer[mask_bool]
    return np.clip(overlay, 0, 255).astype(np.uint8)

def process_image(input_path: str, output_dir: str) -> List[Tuple[str, str]]:
    """
    Main entrypoint. Saves intermediate images into output_dir and returns list of (filename, label).
    Filenames are saved as <base>_<step>.png
    """
    _ensure_dir(output_dir)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    base = os.path.splitext(os.path.basename(input_path))[0]

    # Read image with cv2 (BGR) then convert to RGB
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to read image or unsupported format: " + input_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    outputs = []

    # Step 1: Resize + normalize
    resized = _resize_max_side(img_rgb, MAX_SIDE)
    fname = f"{base}_resized.png"
    _save_rgb(resized, os.path.join(output_dir, fname))
    outputs.append((fname, "Resized"))

    # Step 2: White balance (gray-world)
    wb = gray_world_white_balance(resized)
    fname = f"{base}_wb.png"
    _save_rgb(wb, os.path.join(output_dir, fname))
    outputs.append((fname, "White Balanced"))

    # Step 3: LAB -> CLAHE on L-channel
    clahe_rgb = clahe_on_l_channel(wb, clip_limit=CLAHE_CLIP)
    fname = f"{base}_lab_clahe.png"
    _save_rgb(clahe_rgb, os.path.join(output_dir, fname))
    outputs.append((fname, "CLAHE (L-channel)"))

    # Step 4: Extract A-channel (parasite enhancer)
    a_chan = extract_a_channel(clahe_rgb)
    # normalize a_chan for visualization (stretch contrast)
    a_stretched = exposure.rescale_intensity(a_chan, in_range='image', out_range=(0,255)).astype(np.uint8)
    fname = f"{base}_a_channel.png"
    _save_rgb(cv2.merge([a_stretched, a_stretched, a_stretched]), os.path.join(output_dir, fname))
    outputs.append((fname, "A-channel (parasite enhancer)"))

    # Step 5: Bilateral smoothing (on a-channel to preserve edges)
    bilateral = bilateral_smooth(a_stretched, d=BILATERAL_D, sigma_color=BILATERAL_SIGMA_COLOR, sigma_space=BILATERAL_SIGMA_SPACE)
    fname = f"{base}_bilateral.png"
    _save_rgb(cv2.merge([bilateral, bilateral, bilateral]), os.path.join(output_dir, fname))
    outputs.append((fname, "Bilateral Filtered"))

    # Step 6: Adaptive threshold (on bilateral result)
    # Ensure image is uint8 grayscale
    thr = adaptive_threshold(bilateral, blocksize=ADAPTIVE_BLOCKSIZE, C=ADAPTIVE_C)
    fname = f"{base}_adaptive_thresh.png"
    _save_rgb(thr, os.path.join(output_dir, fname))
    outputs.append((fname, "Adaptive Threshold"))

    # Step 7: Morphological cleanup
    morph = morphological_cleanup(thr, kernel_size=MORPH_KERNEL, min_area=MIN_OBJECT_AREA)
    fname = f"{base}_morph_clean.png"
    _save_rgb(morph, os.path.join(output_dir, fname))
    outputs.append((fname, "Morphological Cleanup"))

    # Step 8: Contour extraction -> final mask (fill + filter)
    final_mask = contours_to_filled_mask(morph, min_area=MIN_OBJECT_AREA)
    fname = f"{base}_final_mask.png"
    _save_rgb(final_mask, os.path.join(output_dir, fname))
    outputs.append((fname, "Final Parasite Mask"))

    # Step 9: Overlay mask on (resized) original for visualization
    overlay = overlay_mask_on_rgb(resized, final_mask, color=(255, 0, 0), alpha=0.45)  # red overlay on RGB
    fname = f"{base}_overlay.png"
    _save_rgb(overlay, os.path.join(output_dir, fname))
    outputs.append((fname, "Overlay on Original"))

    return outputs

# quick local test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test preprocessing pipeline")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("outdir", help="Output directory")
    args = parser.parse_args()
    res = process_image(args.input, args.outdir)
    print("Saved outputs:")
    for fn, label in res:
        print(f" - {fn} : {label}")
