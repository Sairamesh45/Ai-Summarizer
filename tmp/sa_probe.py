import asyncio
from sqlalchemy import text
from app.database import engine

async def main():
    async with engine.connect() as conn:
        r = await conn.execute(text('select 1'))
        print('OK', r.scalar())
    await engine.dispose()

asyncio.run(main())
