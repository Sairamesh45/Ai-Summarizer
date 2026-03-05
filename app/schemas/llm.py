"""
Pydantic schemas for Ollama / LLM integration.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Extraction nested models
# ---------------------------------------------------------------------------


class MedicationItem(BaseModel):
    name: str | None = None
    dosage: str | None = None
    frequency: str | None = None


class LabResultItem(BaseModel):
    test_name: str | None = None
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None  # HIGH | LOW | CRITICAL | NORMAL


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class LlmGenerateRequest(BaseModel):
    """Payload accepted by the /llm/generate endpoint."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32_768,
        description="Text prompt to send to the model.",
    )
    model: str | None = Field(
        default=None,
        description="Override the default Ollama model (e.g. 'llama3', 'mistral').",
    )
    system: str | None = Field(
        default=None,
        max_length=4_096,
        description="Optional system/instruction prompt.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Ollama model options: temperature, top_k, top_p, num_ctx, etc. "
            "See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values"
        ),
    )

    @field_validator("prompt", "system", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v


class LlmSummarizeRequest(BaseModel):
    """Convenience payload for medical-document summarization."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=32_768,
        description="Raw document text to summarize.",
    )
    language: str = Field(
        default="English",
        max_length=64,
        description="Desired output language for the summary.",
    )
    model: str | None = Field(
        default=None, description="Override the default Ollama model."
    )
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


# ---------------------------------------------------------------------------
# Ollama raw response (internal — not exposed to API callers)
# ---------------------------------------------------------------------------


class _OllamaResponse(BaseModel):
    """Mirrors the JSON object returned by POST /api/generate (stream=false)."""

    model: str
    response: str
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None  # nanoseconds
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


# ---------------------------------------------------------------------------
# API responses
# ---------------------------------------------------------------------------


class LlmGenerateResponse(BaseModel):
    """Response returned by the /llm/generate endpoint."""

    text: str = Field(description="Cleaned model output.")
    model: str = Field(description="Ollama model that generated the response.")
    done: bool = Field(description="True when the model finished generating.")
    done_reason: str | None = Field(default=None)
    total_duration_ms: float | None = Field(
        default=None, description="Total inference time in milliseconds."
    )
    prompt_tokens: int | None = Field(default=None)
    completion_tokens: int | None = Field(default=None)


class LlmSummarizeResponse(BaseModel):
    """Response returned by the /llm/summarize endpoint."""

    summary: str = Field(description="AI-generated structured summary.")
    model: str
    total_duration_ms: float | None = None


class LlmExtractRequest(BaseModel):
    """Payload accepted by the /llm/extract endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=32_768,
        description="Raw OCR / document text to extract data from.",
    )
    model: str | None = Field(
        default=None, description="Override the default Ollama model."
    )
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class LlmExtractionResult(BaseModel):
    """Structured medical data extracted from a document."""

    document_date: str | None = None
    diagnoses: list[str] = Field(default_factory=list)

    @field_validator("diagnoses", mode="before")
    @classmethod
    def _normalise_diagnoses(cls, v: Any) -> list[str]:
        """Accept both plain strings and {"name": "..."} dicts from the LLM."""
        if not isinstance(v, list):
            return v
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(item.get("name") or item.get("diagnosis") or str(item))
            else:
                out.append(str(item))
        return out

    medications: list[MedicationItem] = Field(default_factory=list)
    lab_results: list[LabResultItem] = Field(default_factory=list)
    doctor_name: str | None = None
    hospital_name: str | None = None


class LlmExtractResponse(BaseModel):
    """Response returned by the /llm/extract endpoint."""

    data: LlmExtractionResult = Field(
        description="Structured entities extracted from the medical text."
    )
    model: str
    total_duration_ms: float | None = None
    raw_json: str = Field(
        description="The raw JSON string returned by the model (for debugging)."
    )


class LlmPhysicianSummaryRequest(BaseModel):
    """Payload accepted by the /llm/physician-summary endpoint."""

    data: dict[str, Any] = Field(
        ...,
        description=(
            "Structured medical data to summarise — typically the 'data' field "
            "from an LlmExtractResponse, or any dict matching LlmExtractionResult."
        ),
    )
    model: str | None = Field(
        default=None, description="Override the default Ollama model."
    )
    options: dict[str, Any] = Field(default_factory=dict)


class LlmPhysicianSummaryResponse(BaseModel):
    """Response returned by the /llm/physician-summary endpoint."""

    summary: str = Field(description="Concise ~150-word physician-facing summary.")
    word_count: int = Field(description="Approximate word count of the summary.")
    model: str
    total_duration_ms: float | None = None
