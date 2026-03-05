"""
OCR Service — Production-Ready Tesseract Integration
=====================================================

Pipeline per page (each step is independently toggleable):

  raw bytes
    → PIL Image
    → greyscale          always first — all later steps require single-channel
    → remove_shadows     divide out illumination gradient
                         modes: "gaussian" (original) | "morphological" (better
                         for hard shadow edges from phone-camera)
    → contrast           "autocontrast" (global stretch) |
                         "clahe" (tile-local, recommended for medical docs)
    → denoise            "gaussian" (original) | "bilateral" (edge-preserving,
                         better for thin dosage/lab-value text)
    → remove_specks      morphological opening — clears fax dots / printer noise
                         that Tesseract mistakes for punctuation
    → sharpen            unsharp mask — reinforces character stroke edges
    → threshold          "local_mean" (original Sauvola-style) |
                         "sauvola" (true local-stddev Sauvola, better for
                         mixed handwritten + printed pages) |
                         "otsu" (global, fast fallback)
    → deskew             "projection" (original, \u00b115\u00b0) |
                         "hough" (Hough-line, ~10x faster, handles partial skew)
    → border_pad         add white padding so Tesseract never clips edge text
    → upscale            ensure ≥ 1500 px wide
    → table detection    isolate table regions and re-OCR with --psm 6
                         (single uniform block) for better column alignment
    → Tesseract OCR
    → text cleaning

PDF flow:
  bytes → Poppler rasterises each page at target DPI (pdf2image) → same pipeline

All CPU-bound work runs in asyncio.to_thread — the event loop is never blocked.

Accuracy vs. speed guide
────────────────────────
  Maximum accuracy  : all flags True, dpi=400, sauvola threshold, clahe, morphological shadow
  Balanced (default): morphological shadow, clahe, bilateral denoise, remove_specks,
                      sauvola threshold, hough deskew, table_aware
  Fast              : gaussian shadow, autocontrast, gaussian denoise, local_mean threshold,
                      dpi=150, no deskew, no table_aware
"""

from __future__ import annotations

import asyncio
import io
import logging
import platform
import time
from dataclasses import dataclass
from typing import Literal

import pytesseract
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from pdf2image import convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)

from app.config import settings

# ── Logger ────────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── Tesseract binary (Windows only — Linux auto-detects /usr/bin/tesseract) ───
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

# ── MIME types treated as single-frame images ─────────────────────────────────
_IMAGE_MIMES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/tiff",
        "image/tif",
        "image/bmp",
        "image/webp",
        "image/gif",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
#  Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class OcrError(Exception):
    """Base exception for all OCR-related failures."""


class UnsupportedMediaTypeError(OcrError):
    """Raised when the file MIME type cannot be OCR-processed."""


class CorruptFileError(OcrError):
    """Raised when the file bytes are unreadable or corrupt."""


class TesseractUnavailableError(OcrError):
    """Raised when pytesseract cannot find or call the Tesseract binary."""


# ─────────────────────────────────────────────────────────────────────────────
#  Output Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PageResult:
    """OCR result for a single page / frame."""

    page_number: int  # 1-based
    text: str  # cleaned extracted text
    char_count: int  # len(text) after cleaning
    duration_ms: float  # time spent preprocessing + OCR for this page
    skew_angle: float  # degrees corrected (0.0 when deskew is disabled)


@dataclass(slots=True)
class OcrResult:
    """Aggregated OCR result for an entire document."""

    source_type: Literal["image", "pdf"]
    page_count: int
    pages: list[PageResult]
    total_duration_ms: float
    lang: str
    dpi: int

    @property
    def full_text(self) -> str:
        """All page texts joined.  Pages separated by a form-feed (\\f)."""
        return "\f".join(p.text for p in self.pages)

    @property
    def total_char_count(self) -> int:
        return sum(p.char_count for p in self.pages)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "page_count": self.page_count,
            "lang": self.lang,
            "dpi": self.dpi,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_char_count": self.total_char_count,
            "full_text": self.full_text,
            "pages": [
                {
                    "page_number": p.page_number,
                    "char_count": p.char_count,
                    "duration_ms": round(p.duration_ms, 2),
                    "skew_angle": round(p.skew_angle, 2),
                    "text": p.text,
                }
                for p in self.pages
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Image Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class _ImagePreprocessor:
    """
    Stateless image preprocessing pipeline.

    Every method is a pure function: takes a PIL Image, returns a PIL Image
    (plus an angle float for deskew).  All numpy usage is lazy-imported so
    the class works without numpy if deskew / shadow-removal / adaptive-threshold
    are disabled.
    """

    # Minimum width in pixels — images narrower than this are upscaled before OCR
    _MIN_WIDTH = 1500
    # White padding added around the image edges (px)
    _BORDER = 30

    # ── Step 1: greyscale ─────────────────────────────────────────────────────

    @staticmethod
    def to_greyscale(img: Image.Image) -> Image.Image:
        """Convert any mode → greyscale 'L'.  Must always be the first step."""
        return img.convert("L")

    # ── Step 2: shadow / illumination removal ─────────────────────────────────

    @staticmethod
    def remove_shadows(img: Image.Image, radius: int = 60) -> Image.Image:
        """
        Flatfield / illumination correction.

        A very large Gaussian blur approximates the slowly-varying background
        illumination (shadows, uneven flash, phone-camera vignetting).  Dividing
        the original by this estimate normalises brightness across the page.

        Effect: even severely shadowed scans become uniformly lit before
        thresholding, preventing whole regions of text from going solid black.

        Args:
            radius: Blur radius used to estimate illumination.  Should be large
                    enough to cover the largest shadow region (~60 px at 300 DPI).
        """
        try:
            import numpy as np
        except ImportError:
            log.warning("numpy not installed — shadow removal skipped")
            return img

        arr = np.array(img, dtype=np.float32)
        bg = np.array(
            img.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32
        )
        # Divide: pixels in dark areas are boosted; bright areas stay bright
        normalised = (arr / (bg + 1e-6)) * 200.0
        normalised = np.clip(normalised, 0, 255).astype(np.uint8)
        log.debug("Shadow removal (gaussian) applied (radius=%d)", radius)
        return Image.fromarray(normalised, mode="L")

    @staticmethod
    def remove_shadows_morphological(
        img: Image.Image,
        kernel_size: int = 61,
    ) -> Image.Image:
        """
        Background subtraction via morphological closing.

        A large closing kernel fills in all text strokes, producing a
        background-only estimate that captures both gradients AND hard shadow
        edges (phone-camera held at an angle, overhead lamp casting a crease).
        Dividing the original by this estimate flattens both.

        Superior to the Gaussian method when the shadow boundary is sharp rather
        than gradual.

        Args:
            kernel_size: Must be larger than the widest text stroke to avoid
                         including text in the background estimate (~61 at 300 DPI).

        Falls back to Gaussian removal when opencv is not installed.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            log.warning(
                "opencv not installed — morphological shadow removal falling back to Gaussian"
            )
            return _ImagePreprocessor.remove_shadows(img)

        arr = np.array(img, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        background = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
        normalised = cv2.divide(arr, background, scale=255.0)
        log.debug("Shadow removal (morphological) applied (kernel=%d)", kernel_size)
        return Image.fromarray(normalised)

    # ── Step 3: auto-contrast ─────────────────────────────────────────────────

    @staticmethod
    def auto_contrast(img: Image.Image) -> Image.Image:
        """
        Stretch the histogram so the darkest pixel → 0 and brightest → 255.
        The 1% cutoff clips outlier pixels (dust, sensor hot-spots) that would
        otherwise prevent proper stretching.
        """
        return ImageOps.autocontrast(img, cutoff=1)

    @staticmethod
    def clahe(
        img: Image.Image,
        clip_limit: float = 2.0,
        tile_size: int = 8,
    ) -> Image.Image:
        """
        CLAHE — Contrast Limited Adaptive Histogram Equalization.

        Divides the image into a grid of ``tile_size × tile_size`` tiles and
        equalises each tile's histogram independently before interpolating
        between tiles.  The ``clip_limit`` caps the histogram slope to prevent
        noise amplification in uniform regions.

        Why CLAHE outperforms global autocontrast for medical documents:
        • Lab reports, CBC panels, and discharge summaries have bold section
          headers alongside faint body text on the same page
        • Aged / yellowed paper has patchy ink absorption region-to-region
        • Mixed thermal + laser printing varies by section

        Args:
            clip_limit: Contrast cap (slope of equalised CDF).  2.0 is conservative;
                        try 3–4 for very faded scans.
            tile_size:  Grid division count (N×N tiles).  8 = 8×8 grid.

        Falls back to global autocontrast when opencv is not installed.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            log.warning("opencv not installed — CLAHE falling back to autocontrast")
            return ImageOps.autocontrast(img, cutoff=1)

        arr = np.array(img, dtype=np.uint8)
        clahe_filter = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size),
        )
        result = clahe_filter.apply(arr)
        log.debug("CLAHE applied (clip=%.1f tiles=%d)", clip_limit, tile_size)
        return Image.fromarray(result)

    # ── Step 4: denoise ───────────────────────────────────────────────────────

    @staticmethod
    def denoise(img: Image.Image) -> Image.Image:
        """
        Gaussian blur to suppress high-frequency scanner noise and JPEG artefacts.

        A small radius (1–2 px) smooths grain without blurring character strokes.
        Gaussian is preferred over median here because the subsequent sharpen step
        recovers edge definition — the combination denoise→sharpen is more
        effective than median alone.
        """
        return img.filter(ImageFilter.GaussianBlur(radius=1))

    @staticmethod
    def denoise_bilateral(img: Image.Image) -> Image.Image:
        """
        Bilateral filter: smooths noise while hard-preserving edge boundaries.

        Unlike Gaussian (which blurs across edges), bilateral weighs neighbouring
        pixels both by spatial distance AND intensity similarity.  The result is
        noise-free flat regions with razor-sharp stroke edges — critical for
        thin dosage numbers and small-font lab values.

        d=9       : pixel neighbourhood diameter
        sigmaColor: tolerance for intensity similarity (higher = more smoothing
                    across colour steps)
        sigmaSpace: spatial smoothing radius

        Falls back to Gaussian when opencv is not installed.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            log.warning(
                "opencv not installed — bilateral denoise falling back to Gaussian"
            )
            return img.filter(ImageFilter.GaussianBlur(radius=1))

        arr = np.array(img, dtype=np.uint8)
        filtered = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
        return Image.fromarray(filtered)

    @staticmethod
    def remove_specks(img: Image.Image) -> Image.Image:
        """
        Morphological opening removes tiny isolated dark specks.

        Fax artefacts, scanner dust, and thermal-printer noise produce isolated
        1–2 px dark blobs that Tesseract mistakes for commas, periods, or stray
        strokes in numerals.  Opening (erosion → dilation with a 2×2 kernel)
        eliminates specks that are too small to be real glyphs while leaving
        character strokes intact.

        Falls back to a no-op when opencv is not installed.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            log.warning("opencv not installed — remove_specks skipped")
            return img

        arr = np.array(img, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(opened)

    # ── Step 5: sharpen ───────────────────────────────────────────────────────

    @staticmethod
    def sharpen(img: Image.Image) -> Image.Image:
        """
        Unsharp mask — amplifies high-frequency detail (character stroke edges).

        Parameters tuned for OCR use:
          radius=1   : operate at single-pixel edge scale
          percent=180: aggressive boost — text edges become very sharp before
                       binarisation, reducing the chance of merged strokes
          threshold=3: ignore very slow gradients (paper texture, shadows)
        """
        return img.filter(ImageFilter.UnsharpMask(radius=1, percent=180, threshold=3))

    # ── Step 6: adaptive threshold ────────────────────────────────────────────

    @staticmethod
    def adaptive_threshold(
        img: Image.Image,
        block_size: int = 51,
        C: int = 15,
    ) -> Image.Image:
        """
        Local-mean adaptive binarisation (Sauvola-style, numpy implementation).

        For each pixel the threshold is ``local_mean(block_size × block_size) - C``.
        A pixel darker than its local threshold is classified as text (→ 0 / black);
        otherwise background (→ 255 / white).

        Why this beats a fixed global threshold:
        • A global threshold (e.g. pixel > 160 → white) fails when the page has
          shadows, coffee stains, or ageing yellowing — half the page ends up
          solid black and Tesseract reads nothing.
        • Adaptive threshold reacts to *local* illumination so text stays crisp
          even in dark corners of a photographed document.

        Args:
            block_size: Neighbourhood size (must be odd).  Larger = handles
                        bigger illumination gradients; smaller = more sensitive
                        to local contrast.  51 is a good default at 300 DPI.
            C:          Constant subtracted from local mean.  Higher values keep
                        more pixels as background (whiter page); lower values
                        keep more as text.  15 works well for most scans.
        """
        try:
            import numpy as np
        except ImportError:
            log.warning("numpy not installed — falling back to global threshold")
            return img.point(lambda px: 255 if px > 160 else 0, "1").convert("L")

        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1

        arr = np.array(img, dtype=np.float32)
        # Box blur gives local mean cheaply
        local_mean = np.array(
            img.filter(ImageFilter.BoxBlur(block_size // 2)), dtype=np.float32
        )
        # Text pixels are darker than their local neighbourhood by at least C
        binary = np.where(arr < local_mean - C, 0, 255).astype(np.uint8)
        log.debug(
            "Adaptive threshold: block=%d C=%d text_fraction=%.2f%%",
            block_size,
            C,
            100.0 * (binary == 0).sum() / binary.size,
        )
        return Image.fromarray(binary, mode="L")

    @staticmethod
    def sauvola_threshold(
        img: Image.Image,
        window_size: int = 51,
        k: float = 0.2,
        R: float = 128.0,
    ) -> Image.Image:
        """
        True Sauvola binarisation: T(x,y) = mean × (1 + k × (std/R − 1))

        Incorporates both local mean AND local standard deviation, making it
        superior to plain local-mean thresholding when a page mixes:
        • Printed text (uniform ink, high std dev vs. background)
        • Handwritten annotations (variable pressure, lower contrast)
        • Rubber stamps with faded edges
        • Yellowed / ageing paper with patchy ink absorption

        Args:
            window_size: Local neighbourhood (odd integer).  Larger handles bigger
                         illumination gradients; 51 px is a good default at 300 DPI.
            k:           Controls sensitivity.  Original paper default = 0.2.
                         Increase to 0.5 if printed text becomes fragmented.
            R:           Dynamic range of std dev.  128 = half the 8-bit range;
                         matches the original Sauvola paper assumption.
        """
        try:
            import numpy as np
        except ImportError:
            log.warning("numpy not installed — Sauvola falling back to local-mean")
            return _ImagePreprocessor.adaptive_threshold(img)

        if window_size % 2 == 0:
            window_size += 1
        half = window_size // 2

        arr = np.array(img, dtype=np.float64)
        mean = np.array(img.filter(ImageFilter.BoxBlur(half)), dtype=np.float64)

        # E[X²] via squaring then blurring
        sq_img = img.point(lambda p: min(p * p, 65535))
        mean_sq = np.array(sq_img.filter(ImageFilter.BoxBlur(half)), dtype=np.float64)
        std = np.sqrt(np.clip(mean_sq - mean**2, 0, None))

        threshold = mean * (1.0 + k * (std / R - 1.0))
        binary = np.where(arr < threshold, 0, 255).astype(np.uint8)
        log.debug(
            "Sauvola threshold: window=%d k=%.2f text_fraction=%.2f%%",
            window_size,
            k,
            100.0 * (binary == 0).sum() / binary.size,
        )
        return Image.fromarray(binary, mode="L")

    @staticmethod
    def otsu_threshold(img: Image.Image) -> Image.Image:
        """
        Otsu’s global binarisation — fastest threshold option.

        Automatically finds the histogram valley that best separates foreground
        (text) from background.  Works well for clean, uniformly-lit scans but
        degrades with shadows or uneven illumination.  Use as a fast fallback.

        Requires opencv.  Falls back to a fixed 160-threshold without it.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return img.point(lambda px: 255 if px > 160 else 0, "1").convert("L")

        arr = np.array(img, dtype=np.uint8)
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary)

    # ── Step 7: deskew ────────────────────────────────────────────────────────

    @staticmethod
    def deskew(img: Image.Image, max_angle: float = 15.0) -> tuple[Image.Image, float]:
        """
        Estimate and correct document tilt.

        Strategy: projection profile method.
        The image is rotated in 0.5° increments across ±max_angle.  At the
        correct orientation, horizontal text creates sharp periodic peaks in the
        row-sum histogram → maximum row-sum variance indicates the best angle.

        Args:
            max_angle: Search range (degrees).  15° covers almost all real-world
                       document tilt; wider ranges slow down the search linearly.

        Returns:
            ``(corrected_image, angle_degrees)``
        """
        try:
            import numpy as np
        except ImportError:
            log.warning("numpy not installed — deskew skipped")
            return img, 0.0

        best_angle = 0.0
        best_var = -1.0

        angle = -max_angle
        while angle <= max_angle:
            rotated = img.rotate(angle, expand=True, fillcolor=255)
            arr = np.array(rotated, dtype=np.float32)
            variance = float(np.var(arr.sum(axis=1)))
            if variance > best_var:
                best_var = variance
                best_angle = angle
            angle = round(angle + 0.5, 1)

        if abs(best_angle) < 0.1:
            return img, 0.0

        corrected = img.rotate(best_angle, expand=True, fillcolor=255)
        log.debug("Deskew corrected %.2f°", best_angle)
        return corrected, best_angle

    @staticmethod
    def deskew_hough(
        img: Image.Image,
        max_angle: float = 15.0,
    ) -> tuple[Image.Image, float]:
        """
        Hough-line deskewing — ~10× faster than projection profile.

        Uses Canny edge detection followed by Probabilistic Hough Transform to
        find dominant line angles, then rotates by the median of qualifying angles.

        Advantages over projection-profile method:
        • O(edges) instead of O(n_angles × W × H) — ~50 ms vs. ~500 ms per page
        • Robust to partial-page skew (e.g. a rotated table block in an otherwise
          straight document) because it uses median angle, not global maximum
        • minLineLength filter ignores short diagonal table borders that would
          bias the projection method

        Falls back to projection-profile deskew when opencv is not installed.

        Args:
            max_angle: Only angles within this range are accepted as text lines.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            log.warning("opencv not installed — falling back to projection deskew")
            return _ImagePreprocessor.deskew(img, max_angle)

        arr = np.array(img, dtype=np.uint8)
        edges = cv2.Canny(arr, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=img.width // 4,
            maxLineGap=20,
        )
        if lines is None:
            return img, 0.0

        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
            if abs(angle) < max_angle:
                angles.append(angle)

        if not angles:
            return img, 0.0

        best_angle = float(np.median(angles))
        if abs(best_angle) < 0.3:
            return img, 0.0

        corrected = img.rotate(-best_angle, expand=True, fillcolor=255)
        log.debug("Hough deskew corrected %.2f°", -best_angle)
        return corrected, -best_angle

    # ── Step 8: border padding ────────────────────────────────────────────────

    @staticmethod
    def add_border(img: Image.Image) -> Image.Image:
        """
        Add a white border around the image.

        Tesseract's internal segmentation can miss text that starts within
        a few pixels of the image edge.  A small white margin guarantees all
        text is comfortably inside the recognition area.
        """
        bordered = ImageOps.expand(img, border=_ImagePreprocessor._BORDER, fill=255)
        log.debug("Border padding added (%dpx)", _ImagePreprocessor._BORDER)
        return bordered

    # ── Step 9: upscale ───────────────────────────────────────────────────────

    @staticmethod
    def upscale_if_small(img: Image.Image) -> Image.Image:
        """
        Scale up proportionally when the image is narrower than _MIN_WIDTH px.

        Tesseract accuracy degrades significantly on small images because the
        neural LSTM layers were trained on text with a minimum character height.
        Upscaling ensures character strokes are wide enough to recognise.
        """
        w, h = img.size
        if w < _ImagePreprocessor._MIN_WIDTH:
            scale = _ImagePreprocessor._MIN_WIDTH / w
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
            log.debug("Upscaled %dx%d → %dx%d", w, h, *new_size)
        return img

    # ── Table detection ───────────────────────────────────────────────────────────

    @staticmethod
    def detect_table_regions(
        img: Image.Image,
    ) -> list[tuple[int, int, int, int]]:
        """
        Detect table bounding boxes using horizontal + vertical line morphology.

        Strategy:
        1. Invert and binarise the image so grid lines are white-on-black.
        2. Dilate with a wide flat kernel to isolate horizontal lines.
        3. Dilate with a tall kernel to isolate vertical lines.
        4. Add the two masks — their intersection captures table grid cells.
        5. Find contours of the cell regions, group into table bounding boxes.

        Detected tables are then re-OCR’d with ``--psm 6`` (uniform block) which
        preserves column alignment far better than the default ``--psm 3`` on
        dense numeric tables (CBC results, metabolic panels, medication lists).

        Returns list of ``(x, y, w, h)`` sorted top-to-bottom.  Empty list when
        no tables are detected or opencv is not installed.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            log.debug("opencv not installed — table detection skipped")
            return []

        arr = np.array(img, dtype=np.uint8)
        _, binary = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)

        # Horizontal lines: kernel wider than any character gap (~40 px at 300 DPI)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # Vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Grid = union of horizontal and vertical structure
        grid = cv2.add(h_lines, v_lines)
        dilated = cv2.dilate(grid, np.ones((5, 5), np.uint8), iterations=3)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_table_area = img.width * img.height * 0.01  # ignore tiny blobs
        regions: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_table_area:
                regions.append((x, y, w, h))

        log.debug("Table detection found %d region(s)", len(regions))
        return sorted(regions, key=lambda r: r[1])  # top-to-bottom


# ─────────────────────────────────────────────────────────────────────────────
#  OCR Service
# ─────────────────────────────────────────────────────────────────────────────


class OcrService:
    """
    Production-ready async OCR service powered by Tesseract.

    Supported inputs
    ────────────────
    • Single-frame images:  PNG, JPEG, TIFF, BMP, WebP, GIF
    • Multi-page PDFs:      rasterised via Poppler (pdf2image)

    Async design
    ────────────
    All heavy CPU work (rasterisation + preprocessing + Tesseract) runs inside
    asyncio.to_thread() so FastAPI's event loop is never blocked.

    Example
    ───────
        svc = OcrService(lang="eng", deskew=True)
        result = await svc.extract_from_bytes(pdf_bytes, "application/pdf")
        print(result.full_text)
    """

    # PSM 3 = fully automatic page segmentation; OEM 3 = LSTM engine (default)
    _TESS_CONFIG = "--psm 3 --oem 3"
    _TESS_CONFIG_BLOCK = "--psm 6 --oem 3"  # uniform block — better for tables

    def __init__(
        self,
        lang: str = "eng",
        dpi: int = 300,
        *,
        remove_shadows: bool = True,
        shadow_method: Literal["gaussian", "morphological"] = "morphological",
        auto_contrast: bool = True,
        contrast_method: Literal["autocontrast", "clahe"] = "clahe",
        clahe_clip_limit: float = 2.0,
        denoise: bool = True,
        denoise_method: Literal["gaussian", "bilateral"] = "bilateral",
        remove_specks: bool = True,
        sharpen: bool = True,
        adaptive_threshold: bool = True,
        threshold_method: Literal["local_mean", "sauvola", "otsu"] = "sauvola",
        deskew: bool = False,
        deskew_method: Literal["projection", "hough"] = "hough",
        border_pad: bool = True,
        upscale: bool = True,
        table_aware: bool = True,
    ) -> None:
        """
        Args:
            lang:               Tesseract language code(s), e.g. ``"eng"`` or ``"eng+ara"``.
                                The matching language pack must be installed.
            dpi:                PDF render resolution (ignored for image inputs).
                                150 = fast, 300 = best accuracy/speed balance (default),
                                400+ = small-print and low-quality scans.
            remove_shadows:     Divide out illumination gradient.
            shadow_method:      ``"gaussian"`` — original large-blur divide (smooth gradients).
                                ``"morphological"`` — closing-based estimate; also handles
                                hard shadow edges from phone-camera.
            auto_contrast:      Apply contrast enhancement.
            contrast_method:    ``"autocontrast"`` — global histogram stretch.
                                ``"clahe"`` — tile-local adaptive equalization; recommended
                                for medical docs mixing bold headers with faint body text.
            clahe_clip_limit:   CLAHE contrast cap.  2.0 is conservative; 3–4 for faded scans.
            denoise:            Apply denoising.
            denoise_method:     ``"gaussian"`` — simple blur (fast).
                                ``"bilateral"`` — edge-preserving; better for thin dosage text.
            remove_specks:      Morphological opening to eliminate isolated noise blobs
                                (fax dots, printer dust) that Tesseract reads as punctuation.
            sharpen:            Unsharp mask to reinforce character stroke edges before
                                thresholding.
            adaptive_threshold: Binarise the image before Tesseract.
            threshold_method:   ``"local_mean"`` — original mean-only adaptive threshold.
                                ``"sauvola"`` — mean + std dev; better for mixed
                                handwritten/printed pages.
                                ``"otsu"`` — global, fastest, use for clean uniform scans.
            deskew:             Compensate document tilt up to ±15°.  Adds ~50–500 ms/page.
            deskew_method:      ``"hough"`` — Hough-line transform (~50 ms, default).
                                ``"projection"`` — row-sum variance (~500 ms, more robust for
                                very high-tilt documents).
            border_pad:         Add a white border (30 px) so Tesseract never clips edge text.
            upscale:            Scale up images narrower than 1 500 px before OCR.
            table_aware:        After the main OCR pass, detect table regions and re-OCR
                                each with ``--psm 6`` (uniform block) for better column
                                alignment on lab-result panels and medication lists.
        """
        self._lang = lang
        self._dpi = dpi
        self._remove_shadows = remove_shadows
        self._shadow_method = shadow_method
        self._auto_contrast = auto_contrast
        self._contrast_method = contrast_method
        self._clahe_clip_limit = clahe_clip_limit
        self._denoise = denoise
        self._denoise_method = denoise_method
        self._remove_specks = remove_specks
        self._sharpen = sharpen
        self._adaptive_threshold = adaptive_threshold
        self._threshold_method = threshold_method
        self._deskew = deskew
        self._deskew_method = deskew_method
        self._border_pad = border_pad
        self._upscale = upscale
        self._table_aware = table_aware
        self._pre = _ImagePreprocessor()

        log.info(
            "OcrService ready | lang=%s dpi=%d "
            "[shadows=%s/%s contrast=%s/%s denoise=%s/%s specks=%s "
            "sharpen=%s threshold=%s/%s deskew=%s/%s "
            "border=%s upscale=%s table_aware=%s]",
            lang,
            dpi,
            remove_shadows,
            shadow_method,
            auto_contrast,
            contrast_method,
            denoise,
            denoise_method,
            remove_specks,
            sharpen,
            adaptive_threshold,
            threshold_method,
            deskew,
            deskew_method,
            border_pad,
            upscale,
            table_aware,
        )

    # ── Public async API ──────────────────────────────────────────────────────

    async def extract_from_bytes(
        self,
        file_bytes: bytes,
        content_type: str,
    ) -> OcrResult:
        """
        Route bytes to the correct pipeline based on MIME type.

        Args:
            file_bytes:   Raw file content (PDF or image).
            content_type: MIME type string.

        Returns:
            :class:`OcrResult` with per-page breakdown and combined ``full_text``.

        Raises:
            UnsupportedMediaTypeError: MIME type is not processable.
            CorruptFileError:          Bytes cannot be decoded.
            TesseractUnavailableError: Tesseract binary not found or crashed.
            OcrError:                  Any other OCR-level failure.
        """
        log.info(
            "OCR job started | content_type=%s size_kb=%.1f",
            content_type,
            len(file_bytes) / 1024,
        )
        t0 = time.monotonic()

        if content_type == "application/pdf":
            result = await asyncio.to_thread(self._pipeline_pdf, file_bytes)
        elif content_type in _IMAGE_MIMES or content_type.startswith("image/"):
            result = await asyncio.to_thread(self._pipeline_image, file_bytes)
        else:
            raise UnsupportedMediaTypeError(
                f"Cannot OCR content-type '{content_type}'. "
                "Supported: image/*, application/pdf"
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        log.info(
            "OCR job complete | pages=%d chars=%d total_ms=%.0f",
            result.page_count,
            result.total_char_count,
            elapsed_ms,
        )
        return result

    # ── Sync pipelines (executed inside asyncio.to_thread) ───────────────────

    def _pipeline_image(self, data: bytes) -> OcrResult:
        """Single-frame image → preprocess → Tesseract → OcrResult."""
        try:
            raw = Image.open(io.BytesIO(data))
        except UnidentifiedImageError as exc:
            raise CorruptFileError(f"Cannot decode image: {exc}") from exc

        t0 = time.monotonic()
        processed, skew = self._preprocess(raw)
        text = self._run_tesseract(processed)
        duration_ms = (time.monotonic() - t0) * 1000

        return OcrResult(
            source_type="image",
            page_count=1,
            pages=[
                PageResult(
                    page_number=1,
                    text=text,
                    char_count=len(text),
                    duration_ms=duration_ms,
                    skew_angle=skew,
                )
            ],
            total_duration_ms=duration_ms,
            lang=self._lang,
            dpi=self._dpi,
        )

    def _pipeline_pdf(self, data: bytes) -> OcrResult:
        """Multi-page PDF → rasterise → preprocess each page → Tesseract → OcrResult."""
        log.info("Rasterising PDF at %d DPI …", self._dpi)
        t_total = time.monotonic()

        try:
            images: list[Image.Image] = convert_from_bytes(
                data,
                dpi=self._dpi,
                fmt="png",  # lossless rasterisation
                thread_count=2,  # Poppler parallel page rendering
            )
        except PDFPageCountError as exc:
            raise CorruptFileError(f"PDF has no readable pages: {exc}") from exc
        except (PDFSyntaxError, PDFInfoNotInstalledError) as exc:
            raise CorruptFileError(
                f"PDF is corrupt or Poppler is not installed: {exc}"
            ) from exc
        except Exception as exc:
            raise OcrError(f"PDF rasterisation failed unexpectedly: {exc}") from exc

        log.info("Rasterised %d page(s)", len(images))

        page_results: list[PageResult] = []
        for idx, img in enumerate(images, start=1):
            t_page = time.monotonic()
            log.debug("Processing page %d / %d …", idx, len(images))

            processed, skew = self._preprocess(img)
            text = self._run_tesseract(processed)
            page_ms = (time.monotonic() - t_page) * 1000

            page_results.append(
                PageResult(
                    page_number=idx,
                    text=text,
                    char_count=len(text),
                    duration_ms=page_ms,
                    skew_angle=skew,
                )
            )
            log.debug(
                "Page %d done | chars=%d ms=%.0f skew=%.2f°",
                idx,
                len(text),
                page_ms,
                skew,
            )

        return OcrResult(
            source_type="pdf",
            page_count=len(page_results),
            pages=page_results,
            total_duration_ms=(time.monotonic() - t_total) * 1000,
            lang=self._lang,
            dpi=self._dpi,
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, img: Image.Image) -> tuple[Image.Image, float]:
        """
        Run the full preprocessing pipeline in order.

        Each step is gated by the corresponding constructor flag so callers
        can tune the speed/accuracy tradeoff without subclassing.

        Returns:
            ``(processed_image, skew_angle)`` — skew_angle is 0.0 when deskew
            is disabled.
        """
        # 1. Greyscale — always first
        img = self._pre.to_greyscale(img)

        # 2. Shadow / illumination removal
        if self._remove_shadows:
            if self._shadow_method == "morphological":
                img = self._pre.remove_shadows_morphological(img)
            else:
                img = self._pre.remove_shadows(img)

        # 3. Contrast enhancement
        if self._auto_contrast:
            if self._contrast_method == "clahe":
                img = self._pre.clahe(img, clip_limit=self._clahe_clip_limit)
            else:
                img = self._pre.auto_contrast(img)

        # 4. Denoise (before sharpen — removes noise that sharpen would amplify)
        if self._denoise:
            if self._denoise_method == "bilateral":
                img = self._pre.denoise_bilateral(img)
            else:
                img = self._pre.denoise(img)

        # 4b. Speck removal (after denoising, before sharpening)
        if self._remove_specks:
            img = self._pre.remove_specks(img)

        # 5. Sharpen — boost stroke edges before binarisation
        if self._sharpen:
            img = self._pre.sharpen(img)

        # 6. Binarise
        if self._adaptive_threshold:
            if self._threshold_method == "sauvola":
                img = self._pre.sauvola_threshold(img)
            elif self._threshold_method == "otsu":
                img = self._pre.otsu_threshold(img)
            else:  # local_mean
                img = self._pre.adaptive_threshold(img)

        # 7. Deskew — must run on binarised image for best accuracy
        skew = 0.0
        if self._deskew:
            if self._deskew_method == "hough":
                img, skew = self._pre.deskew_hough(img)
            else:
                img, skew = self._pre.deskew(img)

        # 8. Border padding
        if self._border_pad:
            img = self._pre.add_border(img)

        # 9. Upscale small images last (after all transformations)
        if self._upscale:
            img = self._pre.upscale_if_small(img)

        return img, skew

    # ── Tesseract call ────────────────────────────────────────────────────────

    def _run_tesseract(self, img: Image.Image) -> str:
        """
        Call pytesseract and return cleaned text.

        When ``table_aware=True``, detects table regions in the binarised image
        and re-OCRs each with ``--psm 6`` (uniform block) for better column
        alignment, then splices the table text back into the output tagged with
        ``[TABLE] ... [/TABLE]`` markers for downstream parsing.

        Maps all pytesseract exceptions to typed OcrError subclasses.
        """
        try:
            if self._table_aware:
                text = self._run_tesseract_table_aware(img)
            else:
                raw: str = pytesseract.image_to_string(
                    img,
                    lang=self._lang,
                    config=self._TESS_CONFIG,
                )
                text = self._clean(raw)
        except pytesseract.TesseractNotFoundError as exc:
            raise TesseractUnavailableError(
                f"Tesseract binary not found at "
                f"'{pytesseract.pytesseract.tesseract_cmd}'. "
                "Install Tesseract or set TESSERACT_CMD in .env"
            ) from exc
        except pytesseract.TesseractError as exc:
            raise OcrError(f"Tesseract engine error: {exc}") from exc

        return text

    def _run_tesseract_table_aware(self, img: Image.Image) -> str:
        """
        Table-aware OCR pass.

        1. Detect table bounding boxes.
        2. Run the default psm-3 pass on the full page (gives best context for
           non-tabular regions).
        3. Re-run psm-6 on each table crop and tag the result with
           ``[TABLE] ... [/TABLE]`` so downstream NLP/LLM extraction can
           handle structured data specially.
        """
        table_regions = self._pre.detect_table_regions(img)

        # Always run a full-page pass first
        raw_full = pytesseract.image_to_string(
            img, lang=self._lang, config=self._TESS_CONFIG
        )
        parts: list[str] = [self._clean(raw_full)]

        for x, y, w, h in table_regions:
            try:
                crop = img.crop((x, y, x + w, y + h))
                raw_table = pytesseract.image_to_string(
                    crop, lang=self._lang, config=self._TESS_CONFIG_BLOCK
                )
                table_text = self._clean(raw_table)
                if table_text:
                    parts.append(f"\n[TABLE]\n{table_text}\n[/TABLE]\n")
                    log.debug(
                        "Table region (%d,%d,%d,%d) re-OCR’d: %d chars",
                        x,
                        y,
                        w,
                        h,
                        len(table_text),
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Table region OCR failed (%d,%d,%d,%d): %s", x, y, w, h, exc
                )

        return "\n".join(parts)

    # ── Text cleaning ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean(raw: str) -> str:
        """
        Normalise raw Tesseract output:

        1. Normalise line endings
        2. Strip each line
        3. Drop lines that are pure noise (≤ 1 alphanumeric char and < 4 chars total)
        4. Collapse runs of more than 2 consecutive blank lines → single blank line
        5. Strip the whole block
        """
        lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        cleaned: list[str] = []
        blank_run = 0

        for line in lines:
            stripped = line.strip()

            if not stripped:
                blank_run += 1
                # Allow at most one blank line between paragraphs
                if blank_run <= 1:
                    cleaned.append("")
                continue

            blank_run = 0

            # Discard lines that are effectively noise
            alphanumeric = sum(1 for c in stripped if c.isalnum())
            if alphanumeric < 2 and len(stripped) < 4:
                continue

            cleaned.append(stripped)

        return "\n".join(cleaned).strip()
