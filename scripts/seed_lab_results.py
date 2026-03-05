"""
Insert sample lab_result events for patient 27 to test the Lab Report feature.
Run: .\.venv\Scripts\python.exe scripts/seed_lab_results.py
"""

import asyncio
import uuid
from datetime import date, timedelta


async def main():
    from app.database import AsyncSessionLocal
    from app.models.medical import ExtractedEvent
    from app.services.patient_service import PatientService
    from app.services.lab_reference import enrich_lab_result

    async with AsyncSessionLocal() as db:
        svc = PatientService(db)
        pid = await svc.resolve_patient_id("27")
        print(f"Patient UUID: {pid}")

        # Sample lab results with varying flags
        lab_results = [
            # Abnormal values
            {"test_name": "HbA1c", "value": "7.8", "unit": "%"},
            {"test_name": "Fasting Glucose", "value": "142", "unit": "mg/dL"},
            {"test_name": "Total Cholesterol", "value": "245", "unit": "mg/dL"},
            {"test_name": "LDL", "value": "165", "unit": "mg/dL"},
            {"test_name": "Creatinine", "value": "1.5", "unit": "mg/dL"},
            {"test_name": "Hemoglobin", "value": "10.5", "unit": "g/dL"},
            # Normal values
            {"test_name": "HDL", "value": "52", "unit": "mg/dL"},
            {"test_name": "Triglycerides", "value": "130", "unit": "mg/dL"},
            {"test_name": "Sodium", "value": "140", "unit": "mEq/L"},
            {"test_name": "Potassium", "value": "4.2", "unit": "mEq/L"},
            {"test_name": "TSH", "value": "2.1", "unit": "mIU/L"},
            {"test_name": "WBC", "value": "7.5", "unit": "K/μL"},
            {"test_name": "Platelet Count", "value": "250", "unit": "K/μL"},
            {"test_name": "Vitamin D", "value": "18", "unit": "ng/mL"},
        ]

        today = date.today()
        count = 0
        for i, lab in enumerate(lab_results):
            # Enrich with auto-flagging
            lab = enrich_lab_result(lab)
            print(
                f"  {lab['test_name']:25s} = {lab['value']:>6s} {lab.get('unit', ''):8s}  → {lab.get('flag', '—'):8s}  ref: {lab.get('reference_range', '—')}"
            )

            evt = ExtractedEvent(
                patient_id=pid,
                document_id=None,  # Manual seed — no document
                event_type="lab_result",
                event_date=today - timedelta(days=i),
                event_data=lab,
                confidence_score=0.95,
                ai_model="seed_script",
                is_verified=False,
            )
            db.add(evt)
            count += 1

        await db.commit()
        print(f"\n✅ Inserted {count} lab_result events for patient {pid}")


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    asyncio.run(main())
