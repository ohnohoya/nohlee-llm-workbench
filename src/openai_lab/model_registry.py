from typing import Dict

from .schemas import ModelConfig

def _add_model_entries(
    registry: Dict[str, ModelConfig],
    model_id: str,
    supports_reasoning: bool,
    endpoints: Dict[str, str],
    default_endpoint: str,
    supports_temperature: bool = True,
    reasoning_efforts: tuple[str, ...] = (),
    reasoning_default: str | None = None,
) -> None:
    """
    Adds a single model entry with supported endpoints and defaults.
    """
    registry[model_id] = ModelConfig(
        model_id=model_id,
        endpoints=endpoints,
        default_endpoint=default_endpoint,
        supports_reasoning=supports_reasoning,
        supports_temperature=supports_temperature,
        reasoning_efforts=reasoning_efforts,
        reasoning_default=reasoning_default,
    )


MODEL_REGISTRY: Dict[str, ModelConfig] = {}

RESPONSES_MAX_TOKENS_PARAM = "max_output_tokens"
LEGACY_CHAT_MAX_TOKENS_PARAM = "max_tokens"
CHAT_COMPLETION_MAX_TOKENS_PARAM = "max_completion_tokens"

REASONING_NONE_PLUS = ("none", "minimal", "low", "medium", "high")
REASONING_GPT51 = ("none", "low", "medium", "high")
REASONING_PRE_51 = ("minimal", "low", "medium", "high")
REASONING_GPT5_PRO_ONLY = ("high",)
REASONING_GPT52_PRO = ("medium", "high", "xhigh")
REASONING_XHIGH = ("none", "minimal", "low", "medium", "high", "xhigh")

# GPT-5.2 family
# Temperature is only supported when reasoning_effort="none".
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-5.2",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_XHIGH,
    reasoning_default="none",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-5.2-pro",
    supports_reasoning=True,
    supports_temperature=False,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_GPT52_PRO,
    reasoning_default="medium",
)

# GPT-5.1 family
# Temperature is only supported when reasoning_effort="none".
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-5.1",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_GPT51,
    reasoning_default="none",
)

# GPT-5 family
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-5",
    supports_reasoning=True,
    supports_temperature=False,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-5-mini",
    supports_reasoning=True,
    supports_temperature=False,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-5-nano",
    supports_reasoning=True,
    supports_temperature=False,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-5-pro",
    supports_reasoning=True,
    supports_temperature=False,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_GPT5_PRO_ONLY,
    reasoning_default="high",
)

# GPT-4.1 family (non-reasoning)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-4.1",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-4.1-mini",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-4.1-nano",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)

# GPT-4o family (non-reasoning)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-4o",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-4o-mini",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
# o-series (reasoning models; chat uses max_completion_tokens)
_add_model_entries(
    MODEL_REGISTRY,
    "o1",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "o1-mini",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "o1-pro",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "o3",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "o3-mini",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "o3-pro",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "o3-deep-research",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "o4-mini",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)
_add_model_entries(
    MODEL_REGISTRY,
    "o4-mini-deep-research",
    supports_reasoning=True,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": CHAT_COMPLETION_MAX_TOKENS_PARAM},
    default_endpoint="responses",
    reasoning_efforts=REASONING_PRE_51,
    reasoning_default="medium",
)

# Legacy GPT-4 / GPT-3.5
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-4-turbo",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-4",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-3.5-turbo",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-3.5-turbo-0125",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-3.5-turbo-1106",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"responses": RESPONSES_MAX_TOKENS_PARAM, "chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
_add_model_entries(
    MODEL_REGISTRY,
    "gpt-3.5-turbo-16k-0613",
    supports_reasoning=False,
    supports_temperature=True,
    endpoints={"chat_completions": LEGACY_CHAT_MAX_TOKENS_PARAM},
    default_endpoint="chat_completions",
)
