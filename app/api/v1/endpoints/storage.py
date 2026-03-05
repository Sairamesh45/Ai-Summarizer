from fastapi import APIRouter, File, Query, UploadFile

from app.api.deps import CurrentUser, StorageServiceDep

router = APIRouter(prefix="/storage", tags=["storage"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_file(
    service: StorageServiceDep,
    _: CurrentUser,
    file: UploadFile = File(...),
    folder: str = Query(
        default="uploads", description="Target folder inside the bucket"
    ),
):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        from app.core.exceptions import BadRequestException

        raise BadRequestException("File exceeds the 10 MB limit")

    result = await service.upload(folder, file.filename or "file", content)
    return result


@router.delete("/delete")
async def delete_file(
    path: str,
    service: StorageServiceDep,
    _: CurrentUser,
):
    await service.delete(path)
    return {"detail": "File deleted successfully"}


@router.get("/signed-url")
async def signed_url(
    path: str,
    expires_in: int = Query(default=3600, ge=60, le=86400),
    service: StorageServiceDep = ...,
    _: CurrentUser = ...,
):
    url = await service.get_signed_url(path, expires_in)
    return {"url": url, "expires_in": expires_in}
