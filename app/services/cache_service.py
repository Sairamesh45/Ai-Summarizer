"""
In-Process TTL Cache Service
=============================

A lightweight, asyncio-safe in-process cache with per-entry TTL.

Design decisions
----------------
* **In-memory / single-process** — appropriate for a single Uvicorn worker.
  If you scale to multiple workers (Gunicorn + Uvicorn workers or
  Kubernetes replicas), replace this with a shared Redis cache and keep
  the same ``CacheService`` interface.

* **asyncio.Lock per key** — prevents cache stampede: if two requests for
  the same patient arrive simultaneously, only ONE triggers the expensive
  LLM call; the other awaits the lock and then reads the cached value.

* **Generic type annotations** — callers pass ``value_type`` for clarity
  but the cache stores/returns ``Any`` — serialisation is the caller's job.

Usage::

    cache = SummaryCache()          # one module-level singleton

    hit = await cache.get("patient-uuid")
    if hit is None:
        value = await expensive_operation()
        await cache.set("patient-uuid", value)
    else:
        value = hit
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float  # monotonic timestamp


class TtlCache:
    """
    Asyncio-safe TTL cache with optional per-key stampede protection.

    Parameters
    ----------
    ttl:
        Default time-to-live in seconds for every entry.
        Overridable per ``set()`` call.
    max_size:
        Soft cap on number of entries.  When exceeded the oldest
        ``max_size // 4`` entries are evicted (LRU-lite cleanup).
    """

    def __init__(self, ttl: float = 86_400, max_size: int = 4_096) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._store: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()  # global write lock (fast)
        self._key_locks: dict[str, asyncio.Lock] = {}  # per-key locks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` if missing / expired."""
        async with self._lock:
            entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            await self.delete(key)
            return None
        return entry.value

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store ``value`` under ``key`` with the given TTL (defaults to ``self._ttl``)."""
        expires_at = time.monotonic() + (ttl if ttl is not None else self._ttl)
        async with self._lock:
            self._store[key] = _CacheEntry(value=value, expires_at=expires_at)
            if len(self._store) > self._max_size:
                self._evict()

    async def delete(self, key: str) -> None:
        """Remove a single key (no-op if not present)."""
        async with self._lock:
            self._store.pop(key, None)
            self._key_locks.pop(key, None)

    async def invalidate_prefix(self, prefix: str) -> int:
        """Delete all keys that start with ``prefix``. Returns number of evicted entries."""
        async with self._lock:
            victims = [k for k in self._store if k.startswith(prefix)]
            for k in victims:
                self._store.pop(k, None)
                self._key_locks.pop(k, None)
        if victims:
            log.debug(
                "TtlCache.invalidate_prefix: removed %d keys with prefix='%s'",
                len(victims),
                prefix,
            )
        return len(victims)

    def key_lock(self, key: str) -> asyncio.Lock:
        """
        Return (creating if needed) a per-key ``asyncio.Lock``.

        Callers should acquire the returned lock *before* checking ``get()``
        to avoid a cache stampede::

            async with cache.key_lock(key):
                hit = await cache.get(key)
                if hit is None:
                    value = await expensive()
                    await cache.set(key, value)

        """
        if key not in self._key_locks:
            self._key_locks[key] = asyncio.Lock()
        return self._key_locks[key]

    def stats(self) -> dict[str, int]:
        """Return a snapshot of cache stats (size, expired entries)."""
        now = time.monotonic()
        expired = sum(1 for e in self._store.values() if now > e.expires_at)
        return {"size": len(self._store), "expired": expired}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Evict the oldest ``max_size // 4`` entries by expiry time."""
        n_evict = max(1, self._max_size // 4)
        oldest = sorted(self._store.items(), key=lambda kv: kv[1].expires_at)[:n_evict]
        for k, _ in oldest:
            self._store.pop(k, None)
            self._key_locks.pop(k, None)
        log.debug(
            "TtlCache: evicted %d entries (size was > %d)", n_evict, self._max_size
        )


# ---------------------------------------------------------------------------
# Module-level singleton — imported by the endpoint layer
# ---------------------------------------------------------------------------

# Patient summary cache: one entry per patient_id, TTL from settings
summary_cache: TtlCache = TtlCache(
    ttl=float(settings.summary_cache_ttl_seconds),
    max_size=2_048,
)
