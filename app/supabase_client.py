from supabase import AsyncClient, acreate_client

from app.config import settings

_supabase_client: AsyncClient | None = None


async def get_supabase_client() -> AsyncClient:
    """Return a lazily-initialised Supabase async client."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = await acreate_client(
            settings.supabase_url,
            settings.supabase_service_role_key,  # service-role for backend
        )
    return _supabase_client


async def close_supabase_client() -> None:
    global _supabase_client
    _supabase_client = None
