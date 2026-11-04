"""Token price card used by the evaluation report.

This module stays separate from metrics so settings can construct a price
without importing the evaluation report types.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class Pricing(BaseModel):
    """Explicit per-million-token rates supplied by the operator."""

    model_config = ConfigDict(extra="forbid")

    input_usd_per_million_tokens: Decimal = Field(ge=0)
    output_usd_per_million_tokens: Decimal = Field(ge=0)

    def estimate_usd(self, input_tokens: int, output_tokens: int) -> Decimal:
        cost = (
            Decimal(input_tokens) * self.input_usd_per_million_tokens
            + Decimal(output_tokens) * self.output_usd_per_million_tokens
        ) / Decimal(1_000_000)
        return cost.quantize(Decimal("0.000001"))
