import asyncio
import asyncpg


async def try_conn(label, **kwargs):
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(**kwargs, ssl="require", statement_cache_size=0, timeout=8),
            timeout=10,
        )
        ver = await conn.fetchval("SELECT version()")
        await conn.close()
        print(f"SUCCESS [{label}]: {ver[:50]}")
        return True
    except Exception as e:
        print(f"FAIL    [{label}]: {type(e).__name__}: {str(e)[:100]}")
        return False


async def main():
    pw = "Sairamesh1234#"
    ref = "gvjjemzngxyqidwosxxo"

    tests = [
        # Supavisor session pooler (port 5432) - ap regions
        {
            "label": f"supavisor-session-{r}",
            "host": f"aws-0-{r}.pooler.supabase.com",
            "port": 5432,
            "user": f"postgres.{ref}",
            "password": pw,
            "database": "postgres",
        }
        for r in ["ap-south-1", "ap-southeast-1", "ap-southeast-2"]
    ]

    for t in tests:
        label = t.pop("label")
        if await try_conn(label, **t):
            break


asyncio.run(main())
