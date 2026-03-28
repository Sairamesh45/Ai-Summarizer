import asyncio
import asyncpg

DSNS = [
    'postgresql+asyncpg://postgres.gvjjemzngxyqidwosxxo:Sairamesh1234%23@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres',
    'postgresql://postgres.gvjjemzngxyqidwosxxo:Sairamesh1234%23@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres',
    'postgresql://postgres:Sairamesh1234%23@db.gvjjemzngxyqidwosxxo.supabase.co:5432/postgres',
    'postgresql://postgres:Sairamesh1234%23@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres',
    'postgresql://postgres.gvjjemzngxyqidwosxxo:Sairamesh1234%23@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres',
]

async def test(dsn):
    try:
        conn = await asyncpg.connect(dsn=dsn, ssl='require', statement_cache_size=0, timeout=8)
        await conn.fetchval('select 1')
        await conn.close()
        print('OK   ', dsn)
        return True
    except Exception as e:
        print('FAIL ', type(e).__name__, str(e)[:180], '::', dsn)
        return False

async def main():
    for d in DSNS:
        await test(d)

asyncio.run(main())
