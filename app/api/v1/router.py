from fastapi import APIRouter

from app.api.v1.endpoints import auth, documents, llm, patients, secure_files, storage

router = APIRouter(prefix="/api/v1")
router.include_router(auth.router)
router.include_router(storage.router)
router.include_router(documents.router)
router.include_router(secure_files.router)
router.include_router(llm.router)
router.include_router(patients.router)
