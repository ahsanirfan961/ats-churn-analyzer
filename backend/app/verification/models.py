from typing import Literal

from pydantic import BaseModel


class ClaimVerdict(BaseModel):
    text: str
    verdict: Literal["grounded", "mislabeled", "fabricated"]
    source_tool_call_index: int | None = None
    reason: str | None = None


class FaithfulnessResult(BaseModel):
    claims: list[ClaimVerdict] = []
    parse_error: str | None = None

    @property
    def is_faithful(self) -> bool:
        return all(c.verdict == "grounded" for c in self.claims)

    @property
    def problems(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if c.verdict != "grounded"]
