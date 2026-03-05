"""
Test LLM extraction with realistic lab report text.
"""

import asyncio
import logging

logging.basicConfig(level=logging.WARNING)

LAB_TEXT = """
PATIENT LAB REPORT
Patient: John Doe    Date: 2026-03-01
Dr. Smith    City Hospital

COMPLETE BLOOD COUNT (CBC)
Test                    Result   Reference Range   Flag
WBC                     3.2      4.5-11.0 K/uL     LOW
RBC                     5.1      4.2-5.8 M/uL      Normal
Hemoglobin              14.5     12.0-17.5 g/dL    Normal
Hematocrit              43.2     37.0-52.0 %       Normal
Platelet Count          145      150-400 K/uL      LOW

BASIC METABOLIC PANEL
Glucose                 245      70-100 mg/dL      HIGH
BUN                     18       7-25 mg/dL        Normal  
Creatinine              1.9      0.7-1.3 mg/dL     HIGH
Sodium                  138      136-145 mEq/L     Normal
Potassium               3.5      3.5-5.1 mEq/L     Normal

HbA1c                   8.2      < 5.7 %           HIGH
LDL Cholesterol         182      < 100 mg/dL       HIGH
"""


async def main():
    from app.services.llm_service import LlmService

    print("=== Testing LLM extraction with real lab text ===")
    print(f"Input text length: {len(LAB_TEXT)} chars")

    async with LlmService() as llm:
        result = await llm.extract(LAB_TEXT, model="llama3")

    print(f"\nModel: {result.model}")
    print(f"Duration: {result.total_duration_ms:.0f}ms")
    print(f"\n=== Extracted data ===")
    data = result.data
    print(f"Document date: {data.document_date}")
    print(f"Doctor: {data.doctor_name}")
    print(f"Hospital: {data.hospital_name}")
    print(f"Diagnoses ({len(data.diagnoses)}): {data.diagnoses}")
    print(
        f"Medications ({len(data.medications)}): {[m.name for m in data.medications]}"
    )
    print(f"\nLab Results ({len(data.lab_results)}):")
    for lab in data.lab_results:
        print(
            f"  {lab.test_name}: {lab.value} {lab.unit}  ref={lab.reference_range}  flag={lab.flag}"
        )

    print(f"\n=== Raw JSON ===")
    print(result.raw_json[:1000])


asyncio.run(main())
