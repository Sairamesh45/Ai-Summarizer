FROM python:3.13-slim

# ── System dependencies ───────────────────────────────────────────────────────
# tesseract-ocr:     OCR engine + English language pack
# poppler-utils:     PDF rasterisation (required by pdf2image)
# libgl1:            Pillow dependency on headless Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Add more language packs as needed (uncomment):
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     tesseract-ocr-ara \
#     tesseract-ocr-fra \
#     tesseract-ocr-deu \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
