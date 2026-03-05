"""
Full end-to-end pipeline test: create an image with lab report text,
run it through OCR, then LLM extraction.
"""

import asyncio
import io
import logging

logging.basicConfig(level=logging.INFO)


async def main():
    from PIL import Image, ImageDraw, ImageFont
    from app.services.ocr_service import OcrService
    from app.services.llm_service import LlmService

    # ── Build a simple lab-report image ──────────────────────────────────────
    width, height = 1200, 1400
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
        font_bold = ImageFont.truetype("arialbd.ttf", 28)
        font_title = ImageFont.truetype("arialbd.ttf", 32)
    except IOError:
        # Fallback to default font
        font = ImageDraw.Draw(img).font
        font_bold = font
        font_title = font

    def text(x, y, s, f=None, color=(0, 0, 0)):
        draw.text((x, y), s, fill=color, font=f or font)

    text(100, 50, "CITY HOSPITAL — LABORATORY REPORT", font_title)
    text(100, 100, "Patient: John Doe   DOB: 1975-04-12", font_bold)
    text(100, 140, "Physician: Dr. Smith   Date: 2026-03-01", font_bold)
    text(100, 180, "MRN: 00012345", font_bold)

    text(100, 260, "COMPLETE BLOOD COUNT (CBC)", font_bold)
    text(100, 310, "Test Name               Result    Ref Range      Flag")
    text(
        100,
        350,
        "WBC                     3.2       4.5-11.0 K/uL  LOW",
        color=(180, 0, 0),
    )
    text(100, 390, "RBC                     5.1       4.2-5.8 M/uL   Normal")
    text(100, 430, "Hemoglobin              14.5      12.0-17.5 g/dL Normal")
    text(100, 470, "Hematocrit              43.2      37.0-52.0 %    Normal")
    text(
        100,
        510,
        "Platelet Count          145       150-400 K/uL   LOW",
        color=(180, 0, 0),
    )

    text(100, 590, "BASIC METABOLIC PANEL", font_bold)
    text(100, 640, "Test Name               Result    Ref Range      Flag")
    text(
        100,
        680,
        "Glucose                 245       70-100 mg/dL   HIGH",
        color=(180, 0, 0),
    )
    text(100, 720, "BUN                     18        7-25 mg/dL     Normal")
    text(
        100,
        760,
        "Creatinine              1.9       0.7-1.3 mg/dL  HIGH",
        color=(180, 0, 0),
    )
    text(100, 800, "Sodium                  138       136-145 mEq/L  Normal")
    text(100, 840, "Potassium               3.5       3.5-5.1 mEq/L  Normal")

    text(100, 920, "OTHER", font_bold)
    text(100, 970, "Test Name               Result    Ref Range      Flag")
    text(
        100,
        1010,
        "HbA1c                   8.2       <5.7 %         HIGH",
        color=(180, 0, 0),
    )
    text(
        100,
        1050,
        "LDL Cholesterol         182       <100 mg/dL     HIGH",
        color=(180, 0, 0),
    )

    # Save image to bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(200, 200))
    image_bytes = buf.getvalue()
    print(f"Generated test image: {len(image_bytes)} bytes")

    # ── OCR ──────────────────────────────────────────────────────────────────
    print("\n=== Running OCR ===")
    ocr = OcrService(dpi=200)
    result = await ocr.extract_from_bytes(image_bytes, "image/png")
    print(
        f"OCR extracted {result.total_char_count} chars in {result.total_duration_ms:.0f}ms"
    )
    print("--- OCR text (first 800 chars) ---")
    print(result.full_text[:800])

    # ── LLM extraction ───────────────────────────────────────────────────────
    if result.full_text.strip():
        print("\n=== Running LLM extraction ===")
        async with LlmService() as llm:
            extracted = await llm.extract(result.full_text, model="llama3")
        print(f"LLM done in {extracted.total_duration_ms:.0f}ms")
        data = extracted.data
        print(f"\nLab Results ({len(data.lab_results)}):")
        for lab in data.lab_results:
            flag = (
                f"[{lab.flag}]"
                if lab.flag and lab.flag.lower() not in ("normal", "")
                else ""
            )
            print(
                f"  {lab.test_name}: {lab.value} {lab.unit}  ref={lab.reference_range} {flag}"
            )
    else:
        print("\nWARNING: OCR produced no text — pipeline will extract nothing!")


asyncio.run(main())
