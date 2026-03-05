"""Check what documents and events exist for patient 27."""

import asyncio, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def main():
    from sqlalchemy import select, func
    from app.database import AsyncSessionLocal
    from app.models.medical import ExtractedEvent, Document
    from app.services.patient_service import PatientService

    async with AsyncSessionLocal() as db:
        svc = PatientService(db)
        pid = await svc.resolve_patient_id("27")
        print(f"Patient UUID: {pid}\n")

        # Check documents
        result = await db.execute(
            select(Document)
            .where(Document.patient_id == pid)
            .order_by(Document.created_at.desc())
        )
        docs = list(result.scalars().all())
        print(f"=== Documents ({len(docs)}) ===")
        for d in docs:
            print(
                f"  {d.id}  status={d.status}  type={d.document_type}  title={d.title}  created={d.created_at}"
            )

        # Check all events
        result = await db.execute(
            select(ExtractedEvent)
            .where(ExtractedEvent.patient_id == pid)
            .order_by(ExtractedEvent.created_at.desc())
        )
        events = list(result.scalars().all())
        print(f"\n=== Events ({len(events)}) ===")
        for e in events:
            data_preview = str(e.event_data)[:120] if e.event_data else "None"
            print(
                f"  {e.event_type:20s}  date={e.event_date}  doc={e.document_id}  model={e.ai_model}  data={data_preview}"
            )

        # Count by type
        from collections import Counter

        type_counts = Counter(e.event_type for e in events)
        print(f"\n=== Event type counts ===")
        for t, c in type_counts.most_common():
            print(f"  {t}: {c}")


asyncio.run(main())
