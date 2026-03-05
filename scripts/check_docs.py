import asyncio
import logging

logging.basicConfig(level=logging.WARNING)


async def main():
    from app.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as db:
        r = await db.execute(
            text(
                "SELECT id, status, raw_text, page_count, metadata "
                "FROM documents ORDER BY created_at DESC LIMIT 3"
            )
        )
        rows = r.fetchall()
        for row in rows:
            m = dict(row._mapping)
            raw = m.get("raw_text") or ""
            print(f"Doc {m['id']} status={m['status']} pages={m['page_count']}")
            print(f"  metadata={m.get('metadata')}")
            print(f"  raw_text ({len(raw)} chars): {raw[:400]!r}")
            print()


asyncio.run(main())
