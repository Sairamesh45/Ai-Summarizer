from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# ── Engine ────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    # Supabase Supavisor pooler (transaction mode, port 6543) requires SSL
    # and does NOT support prepared statements.
    connect_args={
        "ssl": "require",
        "statement_cache_size": 0,  # disable prepared stmt caching for pooler
    },
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── Base model ────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Dependency ────────────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            # If a flush/execute failed inside the endpoint and was caught there,
            # the session may already be in a pending-rollback state.
            try:
                await session.commit()
            except Exception:
                await session.rollback()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
