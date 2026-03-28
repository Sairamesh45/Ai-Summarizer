"""
LLM Service — Ollama / LLaMA Integration
=========================================

Wraps the Ollama REST API (POST /api/generate) with:
  • Async httpx client with configurable timeout & retry
  • Structured Pydantic validation of every response
  • Clean text output (strips leading/trailing whitespace and think-tags)
  • Granular custom exceptions for upstream errors vs. timeout vs. bad JSON
  • Context-manager lifecycle so the HTTP connection pool is reused and closed
    cleanly on application shutdown.

Usage (inside a FastAPI dependency or endpoint)::

    async with LlmService() as svc:
        result = await svc.generate("Summarise the following…")

Or share a single instance via a FastAPI lifespan dependency (recommended)::

    svc = LlmService()
    await svc.start()
    ...
    await svc.stop()
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
import uuid
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import settings
from app.schemas.llm import (
    LlmExtractionResult,
    LlmExtractResponse,
    LlmGenerateResponse,
    LlmPhysicianSummaryResponse,
    LlmSummarizeResponse,
    _OllamaResponse,
)
from app.schemas.medical import DoctorSummaryResponse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class LlmServiceError(Exception):
    """Base class for all LLM service errors."""


class LlmConnectionError(LlmServiceError):
    """Ollama is not reachable (connection refused, DNS failure, etc.)."""


class LlmTimeoutError(LlmServiceError):
    """The request to Ollama timed out."""


class LlmResponseError(LlmServiceError):
    """Ollama returned an unexpected HTTP status or malformed JSON."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches <think>…</think> blocks emitted by reasoning models (e.g. DeepSeek-R1)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean_text(raw: str) -> str:
    """Strip reasoning blocks and normalise surrounding whitespace."""
    text = _THINK_RE.sub("", raw)
    return text.strip()


# ---------------------------------------------------------------------------
# Medical summarisation prompt template
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM = textwrap.dedent(
    """\
    You are an expert clinical medical scribe.
    When given raw text from a medical document, produce a concise, structured summary in {language}.
    Use the following sections where applicable:
      • Chief Complaint
      • History of Present Illness
      • Past Medical / Surgical History
      • Medications & Allergies
      • Physical Examination findings
      • Investigations / Labs / Imaging
      • Assessment & Diagnoses
      • Plan & Recommendations
    Be precise and clinical. Do NOT hallucinate or invent information not present in the source text.
    If a section has no relevant information, omit it entirely.
"""
)

_SUMMARIZE_PROMPT = textwrap.dedent(
    """\
    Summarise the following medical document text:

    ---
    {text}
    ---
"""
)


# ---------------------------------------------------------------------------
# Medical information extraction prompt template
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = textwrap.dedent(
    """\
    You are a medical information extraction system.
    Extract structured data from the provided medical text.
    Return ONLY valid JSON — no explanations, no markdown fences, no extra text.
    Use null for any field where the information is not present in the text.
    Do NOT invent or hallucinate values.
"""
)

_EXTRACT_PROMPT = textwrap.dedent(
    """\
    Extract structured data from the following medical text.

    Return ONLY valid JSON matching this exact schema:

    {{
      "document_date": "",
      "diagnoses": ["plain string diagnosis 1", "plain string diagnosis 2"],
      "medications": [
        {{
          "name": "",
          "dosage": "",
          "frequency": ""
        }}
      ],
      "lab_results": [
        {{
          "test_name": "",
          "value": "",
          "unit": "",
          "reference_range": "",
          "flag": "HIGH or LOW or NORMAL or CRITICAL or null"
        }}
      ],
      "doctor_name": "",
      "hospital_name": ""
    }}

    If information is missing, use null.

    Medical Text:
    ----------------
    {text}
    ----------------
"""
)

_EXTRACT_CORRECTION_PROMPT = textwrap.dedent(
    """\
    Your previous output was invalid JSON.
    Return ONLY valid JSON — no explanations, no markdown fences, no extra text.

    Required schema:

    {{
      "document_date": "",
      "diagnoses": ["plain string diagnosis 1"],
      "medications": [{{
        "name": "",
        "dosage": "",
        "frequency": ""
      }}],
      "lab_results": [{{
        "test_name": "",
        "value": "",
        "unit": "",
        "reference_range": "",
        "flag": "HIGH or LOW or NORMAL or null"
      }}],
      "doctor_name": "",
      "hospital_name": ""
    }}

    Your previous (invalid) output:
    ----------------
    {previous_output}
    ----------------

    Correct and rewrite as valid JSON only:
"""
)


# ---------------------------------------------------------------------------
# Physician summary prompt template
# ---------------------------------------------------------------------------

_PHYSICIAN_SUMMARY_SYSTEM = textwrap.dedent(
    """\
    You are a medical summarization assistant writing for physicians.
    Generate a concise, clinically accurate summary using ONLY the data provided.
    Do not add information that is not in the source data.
    Do not guess, infer, or hallucinate values.
    Ignore any field that is null or an empty list.
    Write in clear, professional medical language.
    Target length: approximately 150 words.
"""
)

_PHYSICIAN_SUMMARY_PROMPT = textwrap.dedent(
    """\
    Using ONLY the structured medical data below, generate a concise ≈150-word summary for a physician.
    Do not add information. Do not guess. Ignore null or empty fields.

    Data:
    ----------------
    {structured_json}
    ----------------
"""
)


# ---------------------------------------------------------------------------
# Doctor summary prompt (timeline-based, 150–200 words)
# ---------------------------------------------------------------------------

_DOCTOR_SUMMARY_SYSTEM = textwrap.dedent(
    """\
    You are an expert clinical medical scribe generating physician-facing patient summaries.

    STRICT RULES — read carefully:
    1. Use ONLY the structured event data provided. Do NOT hallucinate or infer any information.
    2. If a category has no data, omit it completely. Do not write "no data available".
    3. Format output as a structured bulleted list. Each bullet is one clinical statement.
    4. Begin IMMEDIATELY with the first bullet point. NO preamble, no introductory sentence,
       no phrases like "Here is a summary", "Based on the data", or "The patient presented".
    5. Use a bullet marker (•) at the start of each line. Target 6–10 concise bullets.
    6. Use standard medical terminology and abbreviations where appropriate.

    BULLET CATEGORIES (only include if data present):
    • Primary Diagnoses / Chief Complaint
    • Active Medications
    • Abnormal Lab Findings (explicitly note HIGH/LOW values with units)
    • Vital Signs / Examination findings
    • Procedures / Interventions
    • Follow-up / Clinical Plan
"""
)
_DOCTOR_SUMMARY_PROMPT = textwrap.dedent(
    """\
    Generate a structured physician-facing bullet-point summary for the patient below.
    Use ONLY the chronological structured clinical event data provided.
    Start IMMEDIATELY with the first bullet point — absolutely no preamble or introductory text.
    Each bullet must be a complete, concise clinical statement.
    Highlight chronic diseases, medication changes, and abnormal lab trends.

    Patient Clinical Events (JSON):
    --------------------------------
    {events_json}
    --------------------------------
"""
)

# Matches common LLM preamble lines before the first bullet point
_PREAMBLE_RE = re.compile(
    r"^[^•\-\*]+?(?:summary|patient|below|provided|following|data)[^•\-\*]*?\n+",
    re.IGNORECASE | re.DOTALL,
)


def _strip_preamble(text: str) -> str:
    """Remove any LLM-generated preamble before the first bullet point."""
    # Find the first line that starts with a bullet marker
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0] in ("•", "-", "*"):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def _parse_bullet_points(text: str) -> list[str]:
    """Extract bullet point strings from bulleted summary text."""
    points: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in ("•", "-", "*"):
            if current:
                points.append(" ".join(current))
                current = []
            current = [stripped.lstrip("•-* ").strip()]
        elif current:
            # Continuation of previous bullet
            current.append(stripped)
    if current:
        points.append(" ".join(current))
    return [p for p in points if p]


# Strips optional markdown code fences the model may wrap JSON in
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Extracts the first JSON object from model output (may be truncated — see _repair_json)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Fallback: grabs everything from the first '{' to end-of-string
_JSON_START_RE = re.compile(r"\{.*", re.DOTALL)


def _repair_json(raw: str) -> str:
    """
    Close any unclosed JSON brackets/braces so ``json.loads`` can parse it.

    llama3 (and similar 8B models) sometimes generate structurally valid JSON
    but forget to emit the final ``}`` when the response is long.  This helper
    detects that case and appends the missing closing tokens.

    The function is intentionally conservative: it only adds characters, never
    removes them, and handles strings + escape sequences correctly.
    """
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in raw:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    if not stack:
        return raw  # already balanced

    # Strip trailing commas before closing (common LLM mistake)
    repaired = raw.rstrip()
    while repaired.endswith(","):
        repaired = repaired[:-1].rstrip()

    repaired += "".join(reversed(stack))
    return repaired


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class LlmService:
    """
    Async service for communicating with a locally running Ollama instance.

    Parameters
    ----------
    base_url:
        Root URL of the Ollama server, e.g. ``http://localhost:11434``.
        Defaults to ``settings.ollama_base_url``.
    model:
        Ollama model tag to use, e.g. ``llama3``, ``mistral``, ``phi3``.
        Defaults to ``settings.ollama_model``.
    timeout:
        Total request timeout in seconds.
        Defaults to ``settings.ollama_timeout``.
    max_retries:
        How many times to retry on transient network errors (NOT on 4xx/5xx).
        Defaults to ``settings.ollama_max_retries``.
    """

    _GENERATE_PATH = "/api/generate"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_model
        self._timeout = timeout if timeout is not None else settings.ollama_timeout
        self._max_retries = (
            max_retries if max_retries is not None else settings.ollama_max_retries
        )
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the shared async HTTP client. Call once at startup."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            log.info(
                "LlmService: HTTP client created (base_url=%s, model=%s)",
                self._base_url,
                self._model,
            )

    async def stop(self) -> None:
        """Close the HTTP client. Call once at shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            log.info("LlmService: HTTP client closed.")

    async def __aenter__(self) -> "LlmService":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "LlmService.start() has not been called. Use 'async with LlmService()' or call start()."
            )
        return self._client

    async def _post(self, payload: dict[str, Any]) -> _OllamaResponse:
        """
        POST to /api/generate with automatic retry on connection errors.
        Returns a validated *_OllamaResponse* or raises a typed exception.
        """
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                log.debug(
                    "LlmService: attempt %d — model=%s prompt_len=%d",
                    attempt + 1,
                    payload.get("model"),
                    len(payload.get("prompt", "")),
                )

                response = await self._http.post(self._GENERATE_PATH, json=payload)
                response.raise_for_status()

            except httpx.TimeoutException as exc:
                log.warning("LlmService: timeout on attempt %d — %s", attempt + 1, exc)
                last_exc = LlmTimeoutError(
                    f"Ollama did not respond within {self._timeout}s "
                    f"(model={payload.get('model')}, attempt={attempt + 1})."
                )
                # Timeouts are not worth retrying — the model is just slow
                raise last_exc from exc

            except httpx.ConnectError as exc:
                log.warning(
                    "LlmService: connection error on attempt %d — %s", attempt + 1, exc
                )
                last_exc = LlmConnectionError(
                    f"Could not connect to Ollama at {self._base_url}. "
                    "Make sure Ollama is running (`ollama serve`)."
                )
                if attempt < self._max_retries:
                    continue
                raise last_exc from exc

            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:500]
                raise LlmResponseError(
                    f"Ollama returned HTTP {exc.response.status_code}: {body}"
                ) from exc

            else:
                # Success — parse the response
                try:
                    return _OllamaResponse.model_validate(response.json())
                except (ValidationError, ValueError) as exc:
                    raise LlmResponseError(
                        f"Ollama response could not be parsed. Raw: {response.text[:300]}"
                    ) from exc

        # Should be unreachable but keeps type-checker happy
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> LlmGenerateResponse:
        """
        Send a prompt to the Ollama model and return a structured response.

        Parameters
        ----------
        prompt:
            The user's text prompt.
        model:
            Override the service's default model for this call.
        system:
            Optional system / instruction prompt prepended to context.
        options:
            Ollama model parameters (temperature, top_p, num_ctx, …).

        Returns
        -------
        LlmGenerateResponse
        """
        payload: dict[str, Any] = {
            "model": model or self._model,
            "prompt": prompt,
            "stream": False,  # wait for the complete response before returning
            **({"system": system} if system else {}),
            **({"options": options} if options else {}),
        }

        raw = await self._post(payload)

        total_ms = (raw.total_duration / 1_000_000) if raw.total_duration else None

        log.info(
            "LlmService.generate: done=True model=%s tokens_in=%s tokens_out=%s duration_ms=%.0f",
            raw.model,
            raw.prompt_eval_count or "?",
            raw.eval_count or "?",
            total_ms or 0,
        )

        return LlmGenerateResponse(
            text=_clean_text(raw.response),
            model=raw.model,
            done=raw.done,
            done_reason=raw.done_reason,
            total_duration_ms=total_ms,
            prompt_tokens=raw.prompt_eval_count,
            completion_tokens=raw.eval_count,
        )

    async def summarize(
        self,
        text: str,
        *,
        language: str = "English",
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> LlmSummarizeResponse:
        """
        Summarise medical document text using a built-in clinical prompt template.

        Parameters
        ----------
        text:
            Raw OCR / document text to summarise.
        language:
            Desired output language for the summary (default: English).
        model:
            Override the service's default model for this call.
        options:
            Ollama model parameters.

        Returns
        -------
        LlmSummarizeResponse
        """
        system = _SUMMARIZE_SYSTEM.format(language=language)
        prompt = _SUMMARIZE_PROMPT.format(text=text)

        result = await self.generate(
            prompt, model=model, system=system, options=options
        )

        return LlmSummarizeResponse(
            summary=result.text,
            model=result.model,
            total_duration_ms=result.total_duration_ms,
        )

    async def extract(
        self,
        text: str,
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        max_json_retries: int | None = None,
    ) -> LlmExtractResponse:
        """
        Extract structured medical entities from raw document text.

        Retry pipeline (up to ``max_json_retries`` correction attempts):

          For each attempt:
            1. Strip ``<think>`` reasoning blocks (via ``generate()``).
            2. Strip Markdown code fences.
            3. Regex-extract the first ``{...}`` JSON object.
            4. ``json.loads`` — parse.
            5. Pydantic-validate against ``LlmExtractionResult``.

          If any step fails the model is re-prompted with a correction
          message that includes its previous invalid output:
          *"Your previous output was invalid JSON. Return ONLY valid JSON."*

        Parameters
        ----------
        text:
            Raw OCR / document text to extract data from.
        model:
            Override the service's default model for this call.
        options:
            Ollama model parameters. Defaults to ``temperature: 0`` for
            deterministic, hallucination-resistant output.
        max_json_retries:
            Maximum number of correction retries *after* the first attempt.
            Defaults to ``self._max_retries``.

        Returns
        -------
        LlmExtractResponse

        Raises
        ------
        LlmResponseError
            When all attempts are exhausted without producing valid JSON.
        """
        _max_json = self._max_retries if max_json_retries is None else max_json_retries
        # num_predict=2048: allows ~12-15 lab results + diagnoses + medications without truncation
        # Ollama's default is ~512 which truncates larger lab reports
        merged_options: dict[str, Any] = {
            "temperature": 0,
            "num_predict": 2048,
            **(options or {}),
        }

        last_raw: str = ""
        last_model: str = model or self._model
        last_duration: float | None = None

        for attempt in range(_max_json + 1):
            # ── Build prompt ──────────────────────────────────────────────
            if attempt == 0:
                prompt = _EXTRACT_PROMPT.format(text=text)
                system = _EXTRACT_SYSTEM
            else:
                log.warning(
                    "LlmService.extract: attempt %d/%d — retrying with correction prompt",
                    attempt + 1,
                    _max_json + 1,
                )
                prompt = _EXTRACT_CORRECTION_PROMPT.format(
                    previous_output=last_raw[:2000]
                )
                system = _EXTRACT_SYSTEM

            # ── Generate ──────────────────────────────────────────────────
            result = await self.generate(
                prompt,
                model=model,
                system=system,
                options=merged_options,
            )
            last_raw = result.text
            last_model = result.model
            last_duration = result.total_duration_ms

            # ── Step 1: strip code fences ─────────────────────────────────
            cleaned = _CODE_FENCE_RE.sub("", last_raw).strip()

            # ── Step 2: extract JSON object ───────────────────────────────
            # Try full `{...}` match first; fall back to everything from `{`
            # to end-of-string (handles llama3's missing-closing-brace quirk)
            json_match = _JSON_OBJECT_RE.search(cleaned)
            if json_match:
                json_str = json_match.group(0)
            else:
                start_match = _JSON_START_RE.search(cleaned)
                if not start_match:
                    log.warning(
                        "LlmService.extract: attempt %d — no JSON object found in output",
                        attempt + 1,
                    )
                    continue
                json_str = start_match.group(0)

            # ── Step 3: json.loads (with auto-repair on failure) ──────────
            try:
                payload = json.loads(json_str)
            except json.JSONDecodeError as parse_exc:
                # Try to repair truncated JSON (missing closing `}` / `]`)
                repaired = _repair_json(json_str)
                try:
                    payload = json.loads(repaired)
                    log.info(
                        "LlmService.extract: attempt %d — repaired truncated JSON successfully",
                        attempt + 1,
                    )
                    json_str = repaired  # store the repaired version
                except json.JSONDecodeError:
                    log.warning(
                        "LlmService.extract: attempt %d — JSON parse error: %s",
                        attempt + 1,
                        parse_exc,
                    )
                    continue

            # ── Step 4: Pydantic validation ───────────────────────────────
            try:
                extraction = LlmExtractionResult.model_validate(payload)
            except Exception as val_exc:  # noqa: BLE001
                log.warning(
                    "LlmService.extract: attempt %d — Pydantic validation failed: %s",
                    attempt + 1,
                    val_exc,
                )
                continue

            # ── Success ───────────────────────────────────────────────────
            log.info(
                "LlmService.extract: model=%s diagnoses=%d medications=%d labs=%d "
                "duration_ms=%.0f attempts=%d",
                last_model,
                len(extraction.diagnoses),
                len(extraction.medications),
                len(extraction.lab_results),
                last_duration or 0,
                attempt + 1,
            )
            return LlmExtractResponse(
                data=extraction,
                model=last_model,
                total_duration_ms=last_duration,
                raw_json=json_str,
            )

        # All attempts exhausted
        raise LlmResponseError(
            f"Model failed to return valid JSON after {_max_json + 1} attempt(s). "
            f"Last raw output (first 500 chars): {last_raw[:500]!r}"
        )

    async def physician_summary(
        self,
        data: "dict[str, Any] | LlmExtractionResult",
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> LlmPhysicianSummaryResponse:
        """
        Generate a concise ≈150-word physician-facing summary from structured
        medical data.

        The model is explicitly instructed to use ONLY the data provided —
        no hallucination, no guessing, no invented values.

        Parameters
        ----------
        data:
            Structured medical data — either a plain ``dict`` (e.g. JSON-parsed
            ``LlmExtractionResult``) or an ``LlmExtractionResult`` instance.
        model:
            Override the service's default model for this call.
        options:
            Ollama model parameters.

        Returns
        -------
        LlmPhysicianSummaryResponse
        """
        # Serialise to pretty JSON so the model sees clear key-value pairs
        if isinstance(data, dict):
            json_str = json.dumps(data, indent=2, default=str)
        else:
            json_str = data.model_dump_json(indent=2)

        prompt = _PHYSICIAN_SUMMARY_PROMPT.format(structured_json=json_str)
        result = await self.generate(
            prompt,
            model=model,
            system=_PHYSICIAN_SUMMARY_SYSTEM,
            options=options,
        )

        summary = result.text
        word_count = len(summary.split())

        log.info(
            "LlmService.physician_summary: model=%s word_count=%d duration_ms=%.0f",
            result.model,
            word_count,
            result.total_duration_ms or 0,
        )

        return LlmPhysicianSummaryResponse(
            summary=summary,
            word_count=word_count,
            model=result.model,
            total_duration_ms=result.total_duration_ms,
        )

    async def doctor_summary(
        self,
        events_json: str,
        *,
        patient_id: "uuid.UUID",
        event_count: int,
        model: str | None = None,
    ) -> DoctorSummaryResponse:
        """
        Generate a 150–200-word physician-facing patient summary from a
        chronological JSON string of structured clinical events.

        Parameters
        ----------
        events_json:
            Pre-serialised JSON string of the patient’s clinical events
            (produced by ``PatientService.get_structured_events_for_summary``).
        patient_id:
            UUID of the patient (for the response envelope).
        event_count:
            Number of events included in the context (for the response).
        model:
            Override the service’s default model for this call.

        Returns
        -------
        DoctorSummaryResponse

        Raises
        ------
        LlmTimeoutError
            When Ollama does not respond within ``self._timeout`` seconds.
        LlmConnectionError
            When Ollama cannot be reached.
        LlmResponseError
            When the Ollama response is malformed.
        """
        from datetime import datetime, timezone

        prompt = _DOCTOR_SUMMARY_PROMPT.format(events_json=events_json)
        result = await self.generate(
            prompt,
            model=model,
            system=_DOCTOR_SUMMARY_SYSTEM,
            # temperature=0.3: allows slight natural phrasing variation
            # while keeping factual content stable
            options={"temperature": 0.3},
        )

        # Strip any preamble the model may have emitted despite instructions
        summary = _strip_preamble(result.text)
        summary_points = _parse_bullet_points(summary)
        word_count = len(summary.split())

        log.info(
            "LlmService.doctor_summary: patient=%s model=%s words=%d bullets=%d duration_ms=%.0f",
            patient_id,
            result.model,
            word_count,
            len(summary_points),
            result.total_duration_ms or 0,
        )

        return DoctorSummaryResponse(
            patient_id=patient_id,
            summary=summary,
            summary_points=summary_points,
            word_count=word_count,
            model=result.model,
            event_count=event_count,
            cached=False,
            generated_at=datetime.now(tz=timezone.utc),
            total_duration_ms=result.total_duration_ms,
        )
