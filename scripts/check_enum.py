import asyncio


async def check():
    from app.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as db:
        r = await db.execute(
            text("SELECT unnest(enum_range(NULL::user_role_enum))::text AS val")
        )
        rows = r.fetchall()
        print("DB user_role_enum values:", [row[0] for row in rows])

        # Also check recent documents
        r2 = await db.execute(
            text(
                "SELECT id, status, document_type, created_at FROM documents ORDER BY created_at DESC LIMIT 5"
            )
        )
        rows2 = r2.fetchall()
        print("Recent documents:")
        for row in rows2:
            print(" ", dict(row._mapping))


asyncio.run(check())
