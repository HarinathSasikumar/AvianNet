"""
resize_and_reformat_figures.py
================================
Resizes and crops whitespace from the 5 methodology figures to exact
compact IEEE half-column / single-column dimensions.

Target specs (300 DPI):
  aviannet_architecture.png   : 7.0 in W  x 1.8 in H  (two-column)
  trifold_xai_framework.png   : 7.0 in W  x 1.7 in H  (two-column)
  gat_graph_construction.png  : 3.3 in W  x 2.0 in H  (single-column)
  mlp_fusion_funnel.png       : 3.3 in W  x 1.6 in H  (single-column, ultra-compact)
  perturbation_protocol.png   : 3.3 in W  x 1.8 in H  (single-column)

All outputs saved at 300 DPI with tight bounding boxes.
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageChops
    print("[OK] Pillow imported.")
except ImportError:
    print("[ERROR] Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────────────────

DPI = 300
FIGURES_DIR = Path("E:/Desktop/avian-dna-classifier/figures")

# (filename, target_width_inches, target_height_inches)
FIGURE_SPECS = [
    ("aviannet_architecture.png",   7.0, 1.8),
    ("trifold_xai_framework.png",   7.0, 1.7),
    ("gat_graph_construction.png",  3.3, 2.0),
    ("mlp_fusion_funnel.png",       3.3, 1.6),
    ("perturbation_protocol.png",   3.3, 1.8),
]

# ── Utility: crop whitespace via PIL getbbox on a white-filled bg ─────────────

def autocrop_whitespace(img: Image.Image, bg_color=(255, 255, 255), tolerance=8) -> Image.Image:
    """
    Crops uniform near-white padding around the image content.
    Works on both RGB and RGBA images.
    """
    rgba = img.convert("RGBA")
    # Create a pure-white reference layer
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    diff = ImageChops.difference(rgba, bg)
    # Collapse to single channel: max across bands
    r, g, b, a = diff.split()
    # Use RGB difference only (alpha may vary for PNGs)
    import numpy as np
    arr = np.array(Image.merge("RGB", (r, g, b)))
    mask = arr.max(axis=2) > tolerance  # True where content differs from white
    if not mask.any():
        return img  # All white — return as-is to avoid 0-size crop
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Add a small padding margin (8 px)
    pad = 8
    rmin = max(0, rmin - pad)
    rmax = min(img.height - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(img.width - 1, cmax + pad)
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


def resize_to_exact(img: Image.Image, target_w_px: int, target_h_px: int) -> Image.Image:
    """
    Resizes the image to exactly (target_w_px, target_h_px).
    Uses LANCZOS for downsampling, BICUBIC for upsampling.
    Preserves aspect ratio by fitting within bounds and padding with white.
    """
    orig_w, orig_h = img.size
    scale = min(target_w_px / orig_w, target_h_px / orig_h)
    scaled_w = int(orig_w * scale)
    scaled_h = int(orig_h * scale)
    resample = Image.LANCZOS if scale < 1.0 else Image.BICUBIC
    resized = img.resize((scaled_w, scaled_h), resample=resample)
    # Paste onto exact canvas with white background
    canvas = Image.new("RGB", (target_w_px, target_h_px), (255, 255, 255))
    offset_x = (target_w_px - scaled_w) // 2
    offset_y = (target_h_px - scaled_h) // 2
    # Handle RGBA -> RGB conversion
    if resized.mode == "RGBA":
        r, g, b, a = resized.split()
        resized_rgb = Image.merge("RGB", (r, g, b))
    else:
        resized_rgb = resized.convert("RGB")
    canvas.paste(resized_rgb, (offset_x, offset_y))
    return canvas


# ── Main Processing Loop ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IEEE Figure Resizer — AvianNet (300 DPI)")
    print("=" * 60)

    if not FIGURES_DIR.exists():
        print(f"[ERROR] Figures directory not found: {FIGURES_DIR}")
        sys.exit(1)

    success_count = 0
    for fname, w_in, h_in in FIGURE_SPECS:
        src_path = FIGURES_DIR / fname
        if not src_path.exists():
            print(f"\n[SKIP]  {fname} — file not found at {src_path}")
            continue

        # Target pixel dimensions at 300 DPI
        target_w_px = int(w_in * DPI)
        target_h_px = int(h_in * DPI)

        print(f"\n[PROC]  {fname}")
        print(f"        Target : {w_in:.1f} in x {h_in:.1f} in  ->  {target_w_px} x {target_h_px} px @ 300 DPI")

        # Load original
        img = Image.open(src_path)
        orig_w, orig_h = img.size
        print(f"        Source : {orig_w} x {orig_h} px")

        # Step 1: Autocrop whitespace
        img_rgb = img.convert("RGB")
        cropped = autocrop_whitespace(img_rgb)
        cw, ch = cropped.size
        print(f"        Cropped: {cw} x {ch} px")

        # Step 2: Resize to exact IEEE target
        final = resize_to_exact(cropped, target_w_px, target_h_px)

        # Step 3: Save at 300 DPI in-place
        final.save(src_path, dpi=(DPI, DPI), optimize=False)
        final_size_kb = src_path.stat().st_size // 1024
        print(f"        Saved  : {src_path.name}  ({final_size_kb} KB)")
        success_count += 1

    print(f"\n{'=' * 60}")
    print(f"  Done. {success_count}/{len(FIGURE_SPECS)} figures processed successfully.")
    print(f"  Output directory: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
