from __future__ import annotations

from pydantic import BaseModel


class RuntimeMetrics(BaseModel):
    latency_ms: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
