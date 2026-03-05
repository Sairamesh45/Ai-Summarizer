import asyncio
import logging

logging.basicConfig(level=logging.WARNING)


async def main():
    from app.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as db:
        r = await db.execute(
            text(
                "SELECT ee.event_type, ee.event_data "
                "FROM extracted_events ee "
                "JOIN patients p ON p.id = ee.patient_id "
                "WHERE p.mrn = '27' ORDER BY ee.created_at DESC LIMIT 20"
            )
        )
        rows = r.fetchall()
        print(f"All events for patient 27: {len(rows)}")
        for row in rows:
            print(f"  {row.event_type}: {row.event_data}")


asyncio.run(main())
