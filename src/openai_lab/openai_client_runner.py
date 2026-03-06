import asyncio
from .prompts import PROMPT_GREETINGS, PROMPT_REASONING, MESSAGES_MULTI_TURN_EXAMPLE
from .openai_utils import run_batch, call_openai_hello

# -------------------------
# Runner helpers
# -------------------------
async def run_hello_models(model_names: list[str]) -> dict:
    return await call_openai_hello(model_names)


if __name__ == "__main__":
    RUN_HELLO_ONLY = False  # False runs the full batch matrix (run_batch) instead of hello-only checks

    MODEL_NAMES = [
        # "gpt-5.2",
        # "gpt-5.1",
        # "gpt-5",
        # "gpt-5-mini",
        # "gpt-5-nano",
        # "gpt-4",
        # "gpt-4.1",
        "gpt-4o",
        # "gpt-3.5-turbo",
    ]

    MAX_TOKENS = 1000

    TEMPERATURES = [
        # 0.0,
        # 0.2,
        # 0.8,
        # 1.0,
        None  # None means do not send temperature; the LLM will use its default temperature
    ]

    API_TYPES = [
        # "auto",  # infer from MODEL_REGISTRY
        "responses",
        # "chat_completions",
    ]

    REASONING_EFFORTS = [
        None,  # None means do not send reasoning effort; the LLM will use its default reasoning behavior
        # "none",  # "none" is different from None above. When "none" is sent, the LLM won't use thinking tokens, while it may still reason.
        # "minimal",
        # "low",
        # "medium",
        # "high"
    ]

    APPLY_PARAM_CORRECTIONS = False  # True = default safety/compat checks; False = send reasoning/temperature as configured

    USER_PROMPTS = [
        {"label": "greetings", "prompt": PROMPT_GREETINGS},
        {"label": "reasoning", "prompt": PROMPT_REASONING},
        {"label": "multi turn", "messages": MESSAGES_MULTI_TURN_EXAMPLE},
    ]

    async def _run_all() -> None:
        if RUN_HELLO_ONLY:
            output = await run_hello_models(MODEL_NAMES)
            import json
            print(json.dumps(output, indent=2))
            return

        await run_batch(
            model_names=MODEL_NAMES,
            max_tokens=MAX_TOKENS,
            temperatures=TEMPERATURES,
            api_types=API_TYPES,
            reasoning_efforts=REASONING_EFFORTS,
            requests=USER_PROMPTS,
            apply_param_corrections=APPLY_PARAM_CORRECTIONS,
            concurrency=8,
            close_client=True,
        )

    asyncio.run(_run_all())
