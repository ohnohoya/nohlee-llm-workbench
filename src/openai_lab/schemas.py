from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True)
class ModelConfig:
    model_id: str                  # API model id (e.g., "gpt-4o")
    endpoints: dict[str, str]      # endpoint -> max tokens param
    default_endpoint: str          # "responses" or "chat_completions"
    supports_reasoning: bool       # whether the model accepts a reasoning arg
    supports_temperature: bool     # whether the model accepts temperature
    reasoning_efforts: tuple[str, ...] = ()
    reasoning_default: str | None = None


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    max_tokens: int
    api_type: str = "auto"
    reasoning_set: bool = False
    reasoning_effort: str | None = None
    temperature: float | None = None
    apply_param_corrections: bool = True


class RequestItem(BaseModel):
    label: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[list[dict]] = None

    @model_validator(mode="after")
    def validate_prompt_or_messages(self) -> "RequestItem":
        has_prompt = isinstance(self.prompt, str) and bool(self.prompt.strip())
        has_messages = isinstance(self.messages, list) and len(self.messages) > 0
        if not has_prompt and not has_messages:
            raise ValueError("Each request item must include either a non-empty prompt or messages.")
        return self


class RunRequest(BaseModel):
    model_names: list[str] = ["gpt-4o"]
    requests: list[RequestItem] = Field(
        ...,
        min_length=1,
        examples=[[{"label": "example_1", "prompt": "hi"}]],
    )
    max_tokens: int = 1000
    temperatures: list[Optional[float]] = [0.0]
    api_types: list[Literal["auto", "responses", "chat_completions"]] = ["responses"]
    reasoning_efforts: list[Optional[str]] = [None]
    apply_param_corrections: bool = True
    concurrency: int = 8
    output_path: Optional[str] = Field(
        default=None,
        examples=["output/llm_results_custom.json"],
    )
    return_results: bool = False


class RunResponse(BaseModel):
    status: Literal["ok"]
    output_path: Optional[str]
    count: int
    results: Optional[list[Any]] = None
