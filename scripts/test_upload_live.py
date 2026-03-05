"""
Test upload with a minimal PDF to check the pipeline end to end.
"""

import asyncio
import httpx
import json

# A minimal valid PDF in bytes
MINIMAL_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer<</Size 4/Root 1 0 R>>
startxref
190
%%EOF"""

LAB_PDF_TEXT = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 200>>
stream
BT /F1 12 Tf 50 750 Td (COMPLETE BLOOD COUNT) Tj
0 -20 Td (WBC: 5.2 K/uL  Reference: 4.5-11.0) Tj
0 -20 Td (RBC: 4.8 M/uL  Reference: 4.2-5.8) Tj
0 -20 Td (Hemoglobin: 14.2 g/dL  Reference: 12.0-17.5) Tj
0 -20 Td (Glucose: 245 mg/dL  Reference: 70-100 HIGH) Tj
0 -20 Td (Creatinine: 1.8 mg/dL  Reference: 0.7-1.3 HIGH) Tj
ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000518 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
600
%%EOF"""


async def main():
    import os, sys

    sys.path.insert(0, ".")

    # Get a token first
    from app.core.security import create_access_token
    from app.models.user import UserRoleEnum

    token_data = {"sub": "27", "role": "doctor", "email": "test@clinic.local"}
    token = create_access_token(token_data)
    print(f"Token: {token[:50]}...")

    async with httpx.AsyncClient(
        base_url="http://localhost:8000", timeout=60
    ) as client:
        # Upload a test lab report PDF
        files = {"file": ("lab_report.pdf", MINIMAL_PDF, "application/pdf")}
        data = {
            "patient_id": "27",
            "document_type": "lab_report",
            "title": "Test Lab Report",
        }
        headers = {"Authorization": f"Bearer {token}"}

        print("\n=== Uploading document ===")
        r = await client.post(
            "/api/v1/documents/upload", files=files, data=data, headers=headers
        )
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text[:500]}")

        if r.status_code == 201:
            doc_id = r.json().get("id")
            print(f"\nDocument created: {doc_id}")
            print("Waiting 5s for pipeline to start...")
            await asyncio.sleep(5)

            # Check lab report
            r2 = await client.get("/api/v1/patients/27/lab-report", headers=headers)
            print(f"\n=== Lab Report ===")
            print(f"Status: {r2.status_code}")
            data2 = r2.json()
            print(f"Total: {data2.get('total')}")
            print(f"Still processing: {data2.get('still_processing')}")
            print(f"Items: {data2.get('items')}")


asyncio.run(main())
