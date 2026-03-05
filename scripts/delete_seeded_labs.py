"""Delete the fake seeded lab_result events (ai_model='seed_script')."""

import asyncio, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def main():
    from sqlalchemy import delete
    from app.database import AsyncSessionLocal
    from app.models.medical import ExtractedEvent

    async with AsyncSessionLocal() as db:
        stmt = delete(ExtractedEvent).where(ExtractedEvent.ai_model == "seed_script")
        result = await db.execute(stmt)
        await db.commit()
        print(f"Deleted {result.rowcount} fake seeded events")


asyncio.run(main())
