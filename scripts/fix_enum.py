import asyncio
import logging

logging.basicConfig(level=logging.WARNING)


async def main():
    from app.config import settings
    import asyncpg

    conn = await asyncpg.connect(
        settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    )
    await conn.execute("ALTER TYPE user_role_enum ADD VALUE IF NOT EXISTS 'patient'")
    rows = await conn.fetch(
        "SELECT unnest(enum_range(NULL::user_role_enum))::text AS val"
    )
    print("DB user_role_enum values:", [r["val"] for r in rows])
    await conn.close()


asyncio.run(main())
