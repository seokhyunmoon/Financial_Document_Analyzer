"""
schemas.py
----------
Shared Pydantic schemas used across graph nodes and services.
"""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class QAResponse(BaseModel):
    """Schema used by the QA generator."""

    answer: str = Field(..., description="Final answer text with citations in-line if applicable.")
    citations: list[int] | None = Field(
        None,
        description="List of numeric citation indexes referenced in the answer.",
    )


class EvalResponse(BaseModel):
    """Schema used by the evaluator."""

    classification: str = Field(
        ...,
        description="Evaluation label such as CORRECT, INCORRECT, DIFFERENT, NO_ANSWER, PARTIALLY_CORRECT.",
    )
    reasoning: str = Field(..., description="Brief justification for the classification.")


class ChunkMetadata(BaseModel):
    """Schema for chunk-level metadata returned by the LLM."""

    summary: str = Field(..., description="2-3 sentence summary of the chunk.")
    keywords: List[str] = Field(..., description="Up to N short keyword phrases.")
