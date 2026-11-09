"""Token-count to LLM service-time mapping.

Neither public dataset records per-call latency, so LLM service time is derived from the *real*
token counts of each call with a linear prefill/decode model::

    S_L = overhead + prefill_per_token * context_tokens + decode_per_token * output_tokens

The coefficients are ASSUMED, not measured (defaults: 0.25 s fixed overhead, 0.2 ms per prompt
token, 25 ms per generated token, i.e. ~5,000 tok/s prefill and 40 tok/s decode, typical of a
GPT-4-class hosted endpoint). What is real is the per-call variation: the step-to-step growth
of the context and the spread of output lengths. Because arrival rates are always set from a
target utilization, scaling all three coefficients by the same factor rescales time only for the
LLM pool; the real-data experiments report a sensitivity check on the decode coefficient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TokenLatencyModel:
    """Linear prefill/decode latency model (seconds)."""

    overhead_s: float = 0.25
    prefill_ms_per_token: float = 0.2
    decode_ms_per_token: float = 25.0

    def __post_init__(self) -> None:
        if self.overhead_s < 0 or self.prefill_ms_per_token < 0 or self.decode_ms_per_token <= 0:
            raise ValueError("latency coefficients must be non-negative (decode strictly positive)")

    def service_time(self, context_tokens: int, output_tokens: int) -> float:
        """Service time of one LLM call in seconds."""
        return (
            self.overhead_s
            + 1e-3 * self.prefill_ms_per_token * context_tokens
            + 1e-3 * self.decode_ms_per_token * output_tokens
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
