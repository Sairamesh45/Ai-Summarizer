import mimetypes
import uuid
from pathlib import PurePosixPath

from supabase import AsyncClient

from app.config import settings
from app.core.exceptions import StorageException


class StorageService:
    def __init__(self, client: AsyncClient) -> None:
        self.client = client
        self.bucket = settings.supabase_storage_bucket

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _unique_path(folder: str, original_filename: str) -> str:
        ext = PurePosixPath(original_filename).suffix
        return f"{folder}/{uuid.uuid4()}{ext}"

    @staticmethod
    def _guess_content_type(filename: str) -> str:
        mime, _ = mimetypes.guess_type(filename)
        return mime or "application/octet-stream"

    # ── Public ────────────────────────────────────────────────────────────────
    async def upload(
        self,
        folder: str,
        filename: str,
        data: bytes,
        *,
        upsert: bool = False,
    ) -> dict:
        """Upload bytes to Supabase Storage.  Returns path + public URL."""
        path = self._unique_path(folder, filename)
        content_type = self._guess_content_type(filename)

        try:
            await self.client.storage.from_(self.bucket).upload(
                path=path,
                file=data,
                file_options={
                    "content-type": content_type,
                    "upsert": str(upsert).lower(),
                },
            )
        except Exception as exc:
            raise StorageException(f"Upload failed: {exc}") from exc

        public_url = self._build_public_url(path)
        return {"path": path, "url": public_url, "content_type": content_type}

    async def delete(self, path: str) -> None:
        """Delete a single file from Supabase Storage."""
        try:
            await self.client.storage.from_(self.bucket).remove([path])
        except Exception as exc:
            raise StorageException(f"Delete failed: {exc}") from exc

    async def upload_private(
        self,
        folder: str,
        filename: str,
        data: bytes,
        content_type: str,
    ) -> dict:
        """Upload to a **private** bucket.
        Returns path + a short-lived signed URL (no public access).
        """
        path = self._unique_path(folder, filename)

        try:
            await self.client.storage.from_(self.bucket).upload(
                path=path,
                file=data,
                file_options={
                    "content-type": content_type,
                    "upsert": "false",
                },
            )
        except Exception as exc:
            raise StorageException(f"Upload failed: {exc}") from exc

        signed_url = await self.get_signed_url(path, expires_in=3600)
        return {
            "path": path,
            "signed_url": signed_url,
            "content_type": content_type,
        }

    async def get_signed_url(self, path: str, expires_in: int = 3600) -> str:
        """Return a time-limited signed download URL."""
        try:
            response = await self.client.storage.from_(self.bucket).create_signed_url(
                path, expires_in
            )
            return response["signedURL"]
        except Exception as exc:
            raise StorageException(f"Signed URL generation failed: {exc}") from exc

    async def get_secure_file_url(self, path: str, ttl: int = 300) -> str:
        """Return a 5-minute (default) signed URL for private bucket access.
        Files are never publicly accessible — every retrieval requires a valid
        signed URL issued to an authenticated, role-checked caller.
        """
        if ttl < 60 or ttl > 3600:
            raise StorageException("ttl must be between 60 and 3600 seconds")
        return await self.get_signed_url(path, expires_in=ttl)

    async def download(self, path: str) -> bytes:
        """Download a file from Supabase Storage and return raw bytes."""
        try:
            response = await self.client.storage.from_(self.bucket).download(path)
            return response
        except Exception as exc:
            raise StorageException(f"Download failed: {exc}") from exc

    def _build_public_url(self, path: str) -> str:
        base = settings.supabase_url.rstrip("/")
        return f"{base}/storage/v1/object/public/{self.bucket}/{path}"
