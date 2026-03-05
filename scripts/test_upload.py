from jose import jwt
import httpx

SECRET = "clinic-backend-super-secret-key-change-this-in-production-2026"
# token with numeric sub to simulate Clinic-Backend
payload = {"sub": 27, "roles": ["doctor"], "exp": 9999999999, "iat": 1709568000}
token = jwt.encode(payload, SECRET, algorithm="HS256")
print("TOKEN:", token)

# tiny valid-ish PDF bytes
pdf = b"%PDF-1.1\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Count 0 >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
files = {"file": ("test.pdf", pdf, "application/pdf")}
fields = {"patient_id": "27", "title": "test upload", "document_type": "other"}

try:
    r = httpx.post(
        "http://127.0.0.1:8000/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
        data=fields,
        timeout=30.0,
    )
    print("STATUS", r.status_code)
    print("TEXT", r.text)
    try:
        print("JSON:", r.json())
    except Exception:
        pass
except Exception as e:
    print("REQUEST ERROR", type(e).__name__, e)
    raise
