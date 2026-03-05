"""
Lab Reference Ranges
====================

Built-in reference ranges for common clinical lab tests.
Used to auto-flag lab results as HIGH / LOW / NORMAL when the
source document doesn't include a reference range or flag.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Reference range database
#  Keys are lowercase canonical test names.
#  Each entry: (low, high, unit, display_range)
# ---------------------------------------------------------------------------

_REFERENCE_RANGES: dict[str, tuple[float, float, str, str]] = {
    # Diabetes / Blood sugar
    "hba1c": (4.0, 5.6, "%", "4.0–5.6 %"),
    "hemoglobin a1c": (4.0, 5.6, "%", "4.0–5.6 %"),
    "fasting glucose": (70, 100, "mg/dL", "70–100 mg/dL"),
    "fasting blood sugar": (70, 100, "mg/dL", "70–100 mg/dL"),
    "fbs": (70, 100, "mg/dL", "70–100 mg/dL"),
    "random glucose": (70, 140, "mg/dL", "70–140 mg/dL"),
    "rbs": (70, 140, "mg/dL", "70–140 mg/dL"),
    "pp glucose": (70, 140, "mg/dL", "70–140 mg/dL"),
    "ppbs": (70, 140, "mg/dL", "70–140 mg/dL"),
    "glucose": (70, 100, "mg/dL", "70–100 mg/dL"),
    # Complete Blood Count (CBC)
    "hemoglobin": (12.0, 17.5, "g/dL", "12.0–17.5 g/dL"),
    "hb": (12.0, 17.5, "g/dL", "12.0–17.5 g/dL"),
    "haemoglobin": (12.0, 17.5, "g/dL", "12.0–17.5 g/dL"),
    "hematocrit": (36.0, 50.0, "%", "36–50 %"),
    "hct": (36.0, 50.0, "%", "36–50 %"),
    "rbc": (4.5, 5.5, "M/μL", "4.5–5.5 M/μL"),
    "rbc count": (4.5, 5.5, "M/μL", "4.5–5.5 M/μL"),
    "wbc": (4.5, 11.0, "K/μL", "4.5–11.0 K/μL"),
    "wbc count": (4.5, 11.0, "K/μL", "4.5–11.0 K/μL"),
    "white blood cells": (4.5, 11.0, "K/μL", "4.5–11.0 K/μL"),
    "platelet count": (150, 400, "K/μL", "150–400 K/μL"),
    "platelets": (150, 400, "K/μL", "150–400 K/μL"),
    "plt": (150, 400, "K/μL", "150–400 K/μL"),
    "mcv": (80.0, 100.0, "fL", "80–100 fL"),
    "mch": (27.0, 33.0, "pg", "27–33 pg"),
    "mchc": (32.0, 36.0, "g/dL", "32–36 g/dL"),
    "rdw": (11.5, 14.5, "%", "11.5–14.5 %"),
    "mpv": (7.5, 11.5, "fL", "7.5–11.5 fL"),
    "esr": (0, 20, "mm/hr", "0–20 mm/hr"),
    # Differential count
    "neutrophils": (40.0, 70.0, "%", "40–70 %"),
    "lymphocytes": (20.0, 40.0, "%", "20–40 %"),
    "monocytes": (2.0, 8.0, "%", "2–8 %"),
    "eosinophils": (1.0, 4.0, "%", "1–4 %"),
    "basophils": (0.0, 1.0, "%", "0–1 %"),
    # Kidney function
    "creatinine": (0.7, 1.3, "mg/dL", "0.7–1.3 mg/dL"),
    "serum creatinine": (0.7, 1.3, "mg/dL", "0.7–1.3 mg/dL"),
    "bun": (7.0, 20.0, "mg/dL", "7–20 mg/dL"),
    "blood urea nitrogen": (7.0, 20.0, "mg/dL", "7–20 mg/dL"),
    "urea": (15.0, 40.0, "mg/dL", "15–40 mg/dL"),
    "uric acid": (3.5, 7.2, "mg/dL", "3.5–7.2 mg/dL"),
    "egfr": (90, 120, "mL/min", "≥90 mL/min"),
    # Liver function
    "sgot": (10, 40, "U/L", "10–40 U/L"),
    "ast": (10, 40, "U/L", "10–40 U/L"),
    "sgpt": (7, 56, "U/L", "7–56 U/L"),
    "alt": (7, 56, "U/L", "7–56 U/L"),
    "alkaline phosphatase": (44, 147, "U/L", "44–147 U/L"),
    "alp": (44, 147, "U/L", "44–147 U/L"),
    "total bilirubin": (0.1, 1.2, "mg/dL", "0.1–1.2 mg/dL"),
    "bilirubin": (0.1, 1.2, "mg/dL", "0.1–1.2 mg/dL"),
    "direct bilirubin": (0.0, 0.3, "mg/dL", "0–0.3 mg/dL"),
    "indirect bilirubin": (0.1, 0.8, "mg/dL", "0.1–0.8 mg/dL"),
    "albumin": (3.5, 5.0, "g/dL", "3.5–5.0 g/dL"),
    "total protein": (6.0, 8.3, "g/dL", "6.0–8.3 g/dL"),
    "globulin": (2.0, 3.5, "g/dL", "2.0–3.5 g/dL"),
    "ggt": (9, 48, "U/L", "9–48 U/L"),
    # Lipid profile
    "total cholesterol": (0, 200, "mg/dL", "<200 mg/dL"),
    "cholesterol": (0, 200, "mg/dL", "<200 mg/dL"),
    "ldl": (0, 100, "mg/dL", "<100 mg/dL"),
    "ldl cholesterol": (0, 100, "mg/dL", "<100 mg/dL"),
    "hdl": (40, 60, "mg/dL", "40–60 mg/dL"),
    "hdl cholesterol": (40, 60, "mg/dL", "40–60 mg/dL"),
    "triglycerides": (0, 150, "mg/dL", "<150 mg/dL"),
    "vldl": (5, 40, "mg/dL", "5–40 mg/dL"),
    # Thyroid
    "tsh": (0.4, 4.0, "mIU/L", "0.4–4.0 mIU/L"),
    "t3": (80, 200, "ng/dL", "80–200 ng/dL"),
    "free t3": (2.3, 4.2, "pg/mL", "2.3–4.2 pg/mL"),
    "t4": (5.0, 12.0, "μg/dL", "5.0–12.0 μg/dL"),
    "free t4": (0.8, 1.8, "ng/dL", "0.8–1.8 ng/dL"),
    # Electrolytes
    "sodium": (136, 145, "mEq/L", "136–145 mEq/L"),
    "potassium": (3.5, 5.0, "mEq/L", "3.5–5.0 mEq/L"),
    "chloride": (98, 106, "mEq/L", "98–106 mEq/L"),
    "calcium": (8.5, 10.5, "mg/dL", "8.5–10.5 mg/dL"),
    "phosphorus": (2.5, 4.5, "mg/dL", "2.5–4.5 mg/dL"),
    "magnesium": (1.7, 2.2, "mg/dL", "1.7–2.2 mg/dL"),
    "bicarbonate": (22, 29, "mEq/L", "22–29 mEq/L"),
    # Iron studies
    "iron": (60, 170, "μg/dL", "60–170 μg/dL"),
    "serum iron": (60, 170, "μg/dL", "60–170 μg/dL"),
    "ferritin": (20, 250, "ng/mL", "20–250 ng/mL"),
    "tibc": (250, 370, "μg/dL", "250–370 μg/dL"),
    # Vitamins
    "vitamin d": (30, 100, "ng/mL", "30–100 ng/mL"),
    "25-hydroxyvitamin d": (30, 100, "ng/mL", "30–100 ng/mL"),
    "vitamin b12": (200, 900, "pg/mL", "200–900 pg/mL"),
    "folate": (2.7, 17.0, "ng/mL", "2.7–17.0 ng/mL"),
    "folic acid": (2.7, 17.0, "ng/mL", "2.7–17.0 ng/mL"),
    # Coagulation
    "pt": (11, 13.5, "sec", "11–13.5 sec"),
    "inr": (0.8, 1.1, "", "0.8–1.1"),
    "aptt": (25, 35, "sec", "25–35 sec"),
    # Cardiac markers
    "troponin": (0, 0.04, "ng/mL", "<0.04 ng/mL"),
    "troponin i": (0, 0.04, "ng/mL", "<0.04 ng/mL"),
    "bnp": (0, 100, "pg/mL", "<100 pg/mL"),
    "ck-mb": (0, 25, "U/L", "<25 U/L"),
    "cpk": (10, 120, "U/L", "10–120 U/L"),
    "ldh": (140, 280, "U/L", "140–280 U/L"),
    "crp": (0, 10, "mg/L", "<10 mg/L"),
    "c-reactive protein": (0, 10, "mg/L", "<10 mg/L"),
    "hs-crp": (0, 3, "mg/L", "<3 mg/L"),
    # Pancreatic
    "amylase": (28, 100, "U/L", "28–100 U/L"),
    "lipase": (0, 160, "U/L", "0–160 U/L"),
    # Urinalysis
    "specific gravity": (1.005, 1.030, "", "1.005–1.030"),
    "urine ph": (4.5, 8.0, "", "4.5–8.0"),
}

# Aliases: map common variations
_ALIASES: dict[str, str] = {
    "hgba1c": "hba1c",
    "glycated hemoglobin": "hba1c",
    "glycosylated hemoglobin": "hba1c",
    "a1c": "hba1c",
    "blood sugar": "glucose",
    "blood glucose": "glucose",
    "serum glucose": "glucose",
    "total wbc": "wbc",
    "total rbc": "rbc",
    "s. creatinine": "creatinine",
    "s.creatinine": "creatinine",
    "blood urea": "urea",
    "s. urea": "urea",
    "aspartate aminotransferase": "ast",
    "alanine aminotransferase": "alt",
    "s. bilirubin": "total bilirubin",
    "serum bilirubin": "total bilirubin",
    "total chol": "total cholesterol",
    "tg": "triglycerides",
    "trigs": "triglycerides",
    "na": "sodium",
    "k": "potassium",
    "cl": "chloride",
    "ca": "calcium",
    "mg": "magnesium",
    "phos": "phosphorus",
    "vit d": "vitamin d",
    "vit b12": "vitamin b12",
    "prothrombin time": "pt",
}


def _normalise_test_name(name: str) -> str:
    """Lowercase and strip common prefixes/suffixes."""
    n = name.strip().lower()
    # Remove common prefixes
    for prefix in ("serum ", "s. ", "s.", "blood ", "plasma "):
        if n.startswith(prefix) and n[len(prefix) :] in _REFERENCE_RANGES:
            return n[len(prefix) :]
    return _ALIASES.get(n, n)


def get_reference_range(test_name: str) -> tuple[float, float, str, str] | None:
    """
    Look up a reference range for the given test name.
    Returns (low, high, unit, display_range) or None if not found.
    """
    key = _normalise_test_name(test_name)
    return _REFERENCE_RANGES.get(key)


def auto_flag(test_name: str, value: float) -> tuple[str, str]:
    """
    Determine flag and reference_range for a lab result value.
    Returns (flag, reference_range) — flag is 'HIGH', 'LOW', or 'NORMAL'.
    Returns ('', '') if no reference range is known.
    """
    ref = get_reference_range(test_name)
    if ref is None:
        return ("", "")

    lo, hi, _unit, display = ref
    if value > hi:
        return ("HIGH", display)
    elif value < lo:
        return ("LOW", display)
    else:
        return ("NORMAL", display)


def _parse_range_string(range_str: str) -> tuple[float | None, float | None]:
    """
    Parse a reference range string into (low, high).

    Handles formats like:
      "70-100 mg/dL"  → (70, 100)
      "4.0–5.6 %"     → (4.0, 5.6)
      "< 200 mg/dL"   → (None, 200)
      "<200"           → (None, 200)
      "> 40 mg/dL"    → (40, None)
      "0.7-1.3 mg/dL" → (0.7, 1.3)
    """
    s = range_str.strip()

    # Range with dash/en-dash/em-dash:  "70-100", "4.0–5.6 %"
    m = re.match(r"([\d.]+)\s*[-–—]\s*([\d.]+)", s)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    # Less-than:  "<200", "< 200 mg/dL"
    m = re.match(r"[<≤]\s*([\d.]+)", s)
    if m:
        try:
            return None, float(m.group(1))
        except ValueError:
            pass

    # Greater-than:  ">40", "> 40 mg/dL"
    m = re.match(r"[>≥]\s*([\d.]+)", s)
    if m:
        try:
            return float(m.group(1)), None
        except ValueError:
            pass

    return None, None


def _flag_from_range(value: float, lo: float | None, hi: float | None) -> str:
    """Return HIGH / LOW / NORMAL given parsed range bounds."""
    if lo is not None and value < lo:
        return "LOW"
    if hi is not None and value > hi:
        return "HIGH"
    if lo is None and hi is None:
        return ""  # couldn't parse → unknown
    return "NORMAL"


def enrich_lab_result(lab: dict[str, Any]) -> dict[str, Any]:
    """
    Add flag and reference_range to a lab result dict.
    Modifies and returns the same dict.

    Priority:
    1. If the document already provides a reference_range, parse it and
       compute the flag from that (the report's own normal range).
    2. If the LLM already set a flag (HIGH/LOW/etc.), keep it.
    3. Otherwise fall back to the built-in reference range database.
    """
    test_name = lab.get("test_name") or lab.get("test") or lab.get("name") or ""
    if not test_name:
        return lab

    # ── Extract numeric value ──
    value_raw = lab.get("value", "")
    numeric: float | None = None
    try:
        numeric = float(str(value_raw).strip())
    except (ValueError, TypeError):
        match = re.search(r"[\d.]+", str(value_raw))
        if match:
            try:
                numeric = float(match.group())
            except ValueError:
                pass

    # ── 1.  Document-provided reference range takes priority ──
    doc_range = lab.get("reference_range") or ""
    if doc_range and numeric is not None:
        lo, hi = _parse_range_string(doc_range)
        flag = _flag_from_range(numeric, lo, hi)
        if flag:
            lab["flag"] = flag
            return lab  # keep document's own reference_range string

    # ── 2.  LLM already set a valid flag → trust it ──
    existing_flag = lab.get("flag")
    if existing_flag and str(existing_flag).upper() in (
        "HIGH",
        "LOW",
        "NORMAL",
        "CRITICAL",
    ):
        lab["flag"] = str(existing_flag).upper()
        # Fill in reference range from DB if the document didn't have one
        if not doc_range:
            ref = get_reference_range(test_name)
            if ref:
                lab["reference_range"] = ref[3]
        return lab

    # ── 3.  Fall back to built-in reference range database ──
    if numeric is not None:
        flag, ref_range = auto_flag(test_name, numeric)
        if flag:
            lab["flag"] = flag
        if ref_range and not doc_range:
            lab["reference_range"] = ref_range

    return lab
