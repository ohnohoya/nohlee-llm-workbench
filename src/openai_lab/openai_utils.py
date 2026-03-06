import os
import asyncio
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Optional, List, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .model_registry import MODEL_REGISTRY
from .schemas import RunConfig

load_dotenv()

# -------------------------
# Global AsyncOpenAI client (create once)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable before running this project.")

LLM_TIME_OUT = 15.0  # seconds; lower this to force timeouts
client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIME_OUT)

LOG_PATH: Optional[str] = None
LOG_FIRST = True


def _init_log(path: str) -> None:
    global LOG_PATH
    global LOG_FIRST
    LOG_PATH = path
    LOG_FIRST = True
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("[\n")


def _log_json_local(events: List[Dict[str, Any]], event: Dict[str, Any], sid: Optional[int]) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    if "sid" not in payload and sid is not None:
        payload["sid"] = sid
    events.append(payload)


def _extract_headers(err: Exception) -> Optional[Dict[str, Any]]:
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return dict(headers)
    except Exception:
        return None


def _extract_response_headers(resp: Any) -> Optional[Dict[str, Any]]:
    response = getattr(resp, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    try:
        return dict(headers)
    except Exception:
        return None


def _filter_warning_headers(headers: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not headers:
        return None
    warning_headers = {k: v for k, v in headers.items() if "warning" in k.lower()}
    return warning_headers or None


def _sanitize_auto_reasoning_params(params: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(params)

    reasoning = sanitized.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") == "auto":
        sanitized.pop("reasoning", None)

    if sanitized.get("reasoning_effort") == "auto":
        sanitized.pop("reasoning_effort", None)

    # "auto" and None both mean "do not send temperature".
    if sanitized.get("temperature") in ("auto", None):
        sanitized.pop("temperature", None)

    return sanitized


def _extract_error_info(err: Exception) -> Dict[str, Any]:
    return {
        "error_type": type(err).__name__,
        "error_message": str(err),
        "status_code": getattr(err, "status_code", None),
        "error_code": getattr(err, "code", None),
        "headers": _extract_headers(err),
        "traceback": traceback.format_exc(),
    }


def _log_run_group(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Any] = {}
    for event in events:
        event_name = event.get("event", "unknown")
        event = {k: v for k, v in event.items() if k != "event"}
        if event_name == "llm_meta":
            meta_type = event.get("meta_type")
            if meta_type == "llm_input":
                event = {k: v for k, v in event.items() if k != "meta_type"}
                grouped["llm_request"] = event
            elif meta_type == "response_metadata":
                event = {k: v for k, v in event.items() if k != "meta_type"}
                llm_response = grouped.setdefault("llm_response", {})
                llm_response["metadata"] = event
            else:
                grouped.setdefault("llm_meta", []).append(event)
        elif event_name == "llm_response":
            llm_response = grouped.setdefault("llm_response", {})
            llm_response["output"] = event.get("output")
            if "ts" in event:
                llm_response["ts"] = event["ts"]
        else:
            grouped[event_name] = event
    if "llm_response" in grouped:
        lr = grouped["llm_response"]
        ordered = {}
        if "output" in lr:
            ordered["output"] = lr["output"]
        if "metadata" in lr:
            ordered["metadata"] = lr["metadata"]
        if "ts" in lr:
            ordered["ts"] = lr["ts"]
        grouped["llm_response"] = ordered
    return grouped


def _finalize_log() -> None:
    if LOG_PATH is None:
        return
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n]\n")


def _write_log_groups(path: str, groups: List[Dict[str, Any]]) -> None:
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=True, indent=2)
        f.write("\n")

# -------------------------
# Helpers to call endpoints
# -------------------------
def _build_endpoint_params(
    cfg: Any,
    chosen_endpoint: str,
    max_tokens: int,
    reasoning_set: bool,
    reasoning_effort: Optional[str],
    temperature: Optional[float],
    model_name: str,
    apply_param_corrections: bool = True,
) -> tuple[Dict[str, Any], Optional[str], Optional[str], Optional[str]]:
    params: Dict[str, Any] = {"model": cfg.model_id}

    if chosen_endpoint not in cfg.endpoints:
        raise ValueError(
            f"Endpoint '{chosen_endpoint}' not supported for model {model_name}. "
            f"Supported: {', '.join(cfg.endpoints.keys())}"
        )
    max_tokens_param = cfg.endpoints[chosen_endpoint]
    params[max_tokens_param] = int(max_tokens)

    # reasoning params are endpoint-dependent
    resolved_effort: Optional[str] = None
    reasoning_warning: Optional[str] = None
    temperature_warning: Optional[str] = None
    if reasoning_set:
        if apply_param_corrections:
            if not cfg.supports_reasoning:
                reasoning_warning = (
                    f"reasoning requested but not supported for model {model_name}; "
                    "ignoring reasoning flag."
                )
            elif reasoning_effort is not None:
                if reasoning_effort == "auto":
                    effort = None
                else:
                    effort = reasoning_effort
                if effort is not None:
                    if cfg.reasoning_efforts and effort not in cfg.reasoning_efforts:
                        allowed = ", ".join(cfg.reasoning_efforts)
                        raise ValueError(
                            f"Invalid reasoning_effort '{effort}' for model {model_name}. "
                            f"Allowed: {allowed}"
                        )
                    if chosen_endpoint == "responses":
                        params["reasoning"] = {"effort": effort}
                    else:
                        params["reasoning_effort"] = effort
                    resolved_effort = effort
        elif reasoning_effort is not None:
            # "auto" means "use model/server default", so do not send a reasoning effort param.
            if reasoning_effort != "auto":
                if chosen_endpoint == "responses":
                    params["reasoning"] = {"effort": reasoning_effort}
                else:
                    params["reasoning_effort"] = reasoning_effort
                resolved_effort = reasoning_effort

    # Effective reasoning for downstream compatibility checks:
    # - None means we are not explicitly sending reasoning effort.
    # - "auto" is treated as None (do not send).
    effective_reasoning_effort = resolved_effort
    if not apply_param_corrections and reasoning_set and reasoning_effort not in (None, "auto"):
        effective_reasoning_effort = reasoning_effort

    if temperature is not None:
        if not apply_param_corrections:
            params["temperature"] = float(temperature)
        elif not cfg.supports_temperature:
            temperature_warning = (
                f"temperature requested but not supported for model {model_name}; "
                "omitting temperature."
            )
        elif (
            cfg.model_id.startswith(("gpt-5.1", "gpt-5.2"))
            and effective_reasoning_effort not in (None, "none")
        ):
            temperature_warning = (
                f"temperature requested for model {model_name} with reasoning_effort "
                f"{effective_reasoning_effort}; omitting temperature."
            )
        else:
            params["temperature"] = float(temperature)

    if temperature_warning:
        print(f"warning: {temperature_warning}")

    return params, resolved_effort, reasoning_warning, temperature_warning

async def _call_responses_api(
    params: Dict[str, Any],
    events: List[Dict[str, Any]],
    sid: Optional[int],
    request_label: Optional[str],
    apply_param_corrections: bool,
    reasoning_set: bool,
    reasoning_effort_requested: Optional[str],
    reasoning_effort_resolved: Optional[str],
    reasoning_warning: Optional[str],
    temperature_warning: Optional[str],
) -> Any:
    params = _sanitize_auto_reasoning_params(params)
    _log_json_local(
        events,
        {
            "event": "llm_meta",
            "meta_type": "llm_input",
            "label": request_label,
            "endpoint": "responses",
            "request_start_ts": datetime.now(timezone.utc).isoformat(),
            "apply_param_corrections": apply_param_corrections,
            "reasoning_set": reasoning_set,
            "reasoning_effort_requested": reasoning_effort_requested,
            "reasoning_effort_resolved": reasoning_effort_resolved,
            "reasoning_warning": reasoning_warning,
            "temperature_warning": temperature_warning,
            "temperature": params.get("temperature"),
            "payload": params,
        },
        sid,
    )
    start = time.perf_counter()
    try:
        resp = await client.responses.create(**params)
    except Exception as err:
        _log_json_local(
            events,
            {
                "event": "llm_meta",
                "meta_type": "error",
                "label": request_label,
                "endpoint": "responses",
                "request": params,
                **_extract_error_info(err),
            },
            sid,
        )
        raise
    elapsed = time.perf_counter() - start

    usage = getattr(resp, "usage", None)
    used_model = getattr(resp, "model", params.get("model"))
    output_text = getattr(resp, "output_text", None)
    output_text_len = len(output_text) if isinstance(output_text, str) else None

    def _to_jsonable(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        if hasattr(value, "__dict__"):
            return {k: _to_jsonable(v) for k, v in vars(value).items()}
        return value

    meta = {
        "endpoint": "responses",
        "model": used_model,
        "elapsed_time_s": round(elapsed, 3),
        "usage": _to_jsonable(usage),
        "output_text_len": output_text_len,
    }
    warning_headers = _filter_warning_headers(_extract_response_headers(resp))
    if warning_headers:
        meta["warning_headers"] = warning_headers
    if hasattr(resp, "model_dump"):
        try:
            meta["response_dump"] = resp.model_dump()
        except Exception:
            pass
    _log_json_local(
        events,
        {"event": "llm_meta", "meta_type": "response_metadata", "label": request_label, **meta},
        sid,
    )

    # Prefer concise text extraction; fallback to raw response
    return output_text if output_text is not None else resp


async def call_responses_raw(params: Dict[str, Any]) -> Any:
    safe_params = _sanitize_auto_reasoning_params(params)
    return await client.responses.create(**safe_params)

async def call_openai_hello(model_names: List[str]) -> Dict[str, Any]:
    if not model_names:
        return {}
    tasks = [call_responses_raw({"model": name, "input": "hello"}) for name in model_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    messages: Dict[str, Any] = {}
    for name, result in zip(model_names, results):
        if isinstance(result, Exception):
            messages[name] = str(result)
            continue
        output_text = getattr(result, "output_text", None)
        messages[name] = output_text if output_text is not None else result
    return messages

async def _call_chat_completions_api(
    params: Dict[str, Any],
    events: List[Dict[str, Any]],
    sid: Optional[int],
    request_label: Optional[str],
    apply_param_corrections: bool,
    reasoning_set: bool,
    reasoning_effort_requested: Optional[str],
    reasoning_effort_resolved: Optional[str],
    reasoning_warning: Optional[str],
    temperature_warning: Optional[str],
) -> Any:
    params = _sanitize_auto_reasoning_params(params)
    _log_json_local(
        events,
        {
            "event": "llm_meta",
            "meta_type": "llm_input",
            "label": request_label,
            "endpoint": "chat_completions",
            "request_start_ts": datetime.now(timezone.utc).isoformat(),
            "apply_param_corrections": apply_param_corrections,
            "reasoning_set": reasoning_set,
            "reasoning_effort_requested": reasoning_effort_requested,
            "reasoning_effort_resolved": reasoning_effort_resolved,
            "reasoning_warning": reasoning_warning,
            "temperature_warning": temperature_warning,
            "temperature": params.get("temperature"),
            "payload": params,
        },
        sid,
    )
    start = time.perf_counter()
    try:
        completion = await client.chat.completions.create(**params)
    except Exception as err:
        _log_json_local(
            events,
            {
                "event": "llm_meta",
                "meta_type": "error",
                "label": request_label,
                "endpoint": "chat_completions",
                "request": params,
                **_extract_error_info(err),
            },
            sid,
        )
        raise
    elapsed = time.perf_counter() - start

    usage = getattr(completion, "usage", None)
    used_model = getattr(completion, "model", params.get("model"))
    try:
        choice0 = completion.choices[0].message.content
    except Exception:
        choice0 = None

    output_text_len = len(choice0) if isinstance(choice0, str) else None
    try:
        finish_reason = completion.choices[0].finish_reason
    except Exception:
        finish_reason = None

    def _to_jsonable(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        if hasattr(value, "__dict__"):
            return {k: _to_jsonable(v) for k, v in vars(value).items()}
        return value

    meta = {
        "endpoint": "chat_completions",
        "model": used_model,
        "elapsed_time_s": round(elapsed, 3),
        "usage": _to_jsonable(usage),
        "finish_reason": finish_reason,
        "output_text_len": output_text_len,
    }
    warning_headers = _filter_warning_headers(_extract_response_headers(completion))
    if warning_headers:
        meta["warning_headers"] = warning_headers
    if hasattr(completion, "model_dump"):
        try:
            meta["response_dump"] = completion.model_dump()
        except Exception:
            pass
    _log_json_local(
        events,
        {"event": "llm_meta", "meta_type": "response_metadata", "label": request_label, **meta},
        sid,
    )

    # typical SDK object: choices -> first -> message -> content
    return choice0 if choice0 is not None else completion


# -------------------------
# Unified generate function using registry
# -------------------------
async def generate_with_model(
    model_name: str,
    prompt: str,
    max_tokens: int,
    messages: Optional[List[Dict[str, str]]] = None,
    api_type: str = "auto",     # "auto" => infer from registry; or "responses" / "chat_completions" to override
    reasoning_set: bool = False,            # toggle on/off
    reasoning_effort: Optional[str] = None,   # if reasoning_set True and supported, what effort to request
    request_label: Optional[str] = None,
    temperature: Optional[float] = None,
    apply_param_corrections: bool = True,
    events: Optional[List[Dict[str, Any]]] = None,
    sid: Optional[int] = None,
) -> Any:
    """
    Build the right params automatically from MODEL_REGISTRY, then dispatch to the right API.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Add it to MODEL_REGISTRY or correct the name.")

    cfg = MODEL_REGISTRY[model_name]
    if api_type not in ("auto", "responses", "chat_completions"):
        raise ValueError("api_type must be 'auto', 'responses', or 'chat_completions'.")

    chosen_endpoint = cfg.default_endpoint if api_type == "auto" else api_type

    params, resolved_effort, reasoning_warning, temperature_warning = _build_endpoint_params(
        cfg=cfg,
        chosen_endpoint=chosen_endpoint,
        max_tokens=max_tokens,
        reasoning_set=reasoning_set,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        model_name=model_name,
        apply_param_corrections=apply_param_corrections,
    )
    reasoning_effort_requested = reasoning_effort

    if chosen_endpoint == "responses":
        if messages is not None:
            params["input"] = messages
        else:
            params["input"] = prompt
        return await _call_responses_api(
            params,
            events=events or [],
            sid=sid,
            request_label=request_label,
            apply_param_corrections=apply_param_corrections,
            reasoning_set=reasoning_set,
            reasoning_effort_requested=reasoning_effort_requested,
            reasoning_effort_resolved=resolved_effort,
            reasoning_warning=reasoning_warning,
            temperature_warning=temperature_warning,
        )
    else:
        # chat_completions -> messages
        if messages is not None:
            params["messages"] = messages
        else:
            params["messages"] = [{"role": "user", "content": prompt}]
        return await _call_chat_completions_api(
            params,
            events=events or [],
            sid=sid,
            request_label=request_label,
            apply_param_corrections=apply_param_corrections,
            reasoning_set=reasoning_set,
            reasoning_effort_requested=reasoning_effort_requested,
            reasoning_effort_resolved=resolved_effort,
            reasoning_warning=reasoning_warning,
            temperature_warning=temperature_warning,
        )


# -------------------------
# Main: parameterized, no global read
# -------------------------
async def run_request_collect(
    run_config: RunConfig,
    req: Dict[str, Any],
    idx: int,
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    model_name = run_config.model_name
    max_tokens = run_config.max_tokens
    api_type = run_config.api_type
    reasoning_set = run_config.reasoning_set
    reasoning_effort = run_config.reasoning_effort
    temperature = run_config.temperature
    apply_param_corrections = run_config.apply_param_corrections

    label = req.get("label", f"request_{idx}")
    sid = req.get("sid", idx)
    _log_json_local(
        events,
        {
            "event": "llm_call_config",
            "sid": sid,
            "model_name": model_name,
            "max_tokens": max_tokens,
            "api_type": api_type,
            "reasoning_set": reasoning_set,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
        },
        sid,
    )
    try:
        out = await generate_with_model(
            model_name=model_name,
            prompt=req.get("prompt", ""),
            messages=req.get("messages"),
            max_tokens=max_tokens,
            api_type=api_type,
            reasoning_set=reasoning_set,
            reasoning_effort=req.get("reasoning_effort", reasoning_effort),
            request_label=label,
            temperature=temperature,
            apply_param_corrections=apply_param_corrections,
            events=events,
            sid=sid,
        )
        _log_json_local(
            events,
            {
                "event": "llm_response",
                "output": out,
            },
            sid,
        )
    except Exception as err:
        _log_json_local(
            events,
            {
                "event": "run_error",
                "label": label,
                "model_name": model_name,
                "api_type": api_type,
                **_extract_error_info(err),
            },
            sid,
        )
        print(f"sid={sid} model={model_name} error={type(err).__name__}: {err}")
    return _log_run_group(events)


async def run_batch(
    *,
    model_names: list[str],
    max_tokens: int,
    temperatures: list[float | None],
    api_types: list[str],
    reasoning_efforts: list[str | None],
    requests: list[dict],
    apply_param_corrections: bool = True,
    concurrency: int = 8,
    output_path: str | None = None,
    close_client: bool = False,
) -> dict:
    ts = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%dT%H%M%S%z")
    sid = 0
    requests_to_run = []
    for model_name in model_names:
        for api_type in api_types:
            for effort in reasoning_efforts:
                reasoning_set = effort is not None
                for temperature in temperatures:
                    run_config = RunConfig(
                        model_name=model_name,
                        max_tokens=max_tokens,
                        api_type=api_type,
                        reasoning_set=reasoning_set,
                        reasoning_effort=effort,
                        temperature=temperature,
                        apply_param_corrections=apply_param_corrections,
                    )
                    for idx, req in enumerate(requests, start=1):
                        sid += 1
                        label = req.get("label", f"request_{idx}")
                        print(
                            f"Submitting a task for sid={sid} model={model_name} api_type={api_type} "
                            f"reasoning_set={reasoning_set} reasoning_effort={effort} "
                            f"temperature={temperature} label={label}"
                        )
                        requests_to_run.append(
                            (run_config, {**req, "label": label, "sid": sid})
                        )

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = [None] * len(requests_to_run)

    async def _runner(i: int, rc: RunConfig, req: dict) -> None:
        async with semaphore:
            results[i] = await run_request_collect(rc, req, i + 1)

    try:
        if output_path:
            out_path = Path(output_path)
            if out_path.parent == Path("."):
                output_path = str(Path("output") / out_path.name)
        tasks = [asyncio.create_task(_runner(i, rc, req)) for i, (rc, req) in enumerate(requests_to_run)]
        await asyncio.gather(*tasks)
        final_output_path = output_path or f"output/llm_results_{ts}.json"
        _write_log_groups(final_output_path, results)
        failed_count = sum(1 for item in results if isinstance(item, dict) and "run_error" in item)
        return {"output_path": final_output_path, "results": results, "failed_count": failed_count}
    finally:
        if close_client:
            await client.close()


if __name__ == "__main__":
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    prompt = " ".join(sys.argv[1:]).strip() or "Say hello in one sentence."

    async def _run() -> None:
        params = {
            "model": "gpt-5.2",
            "max_output_tokens": 2000,
            "input": "do you think a nuclear war will ever take place??",
            "reasoning": {"effort": "medium"},
            # "temperature": 0.0,
        }
        resp = await call_responses_raw(params)
        headers = _extract_response_headers(resp)
        if headers:
            print(f"response_headers={headers}")
        usage = getattr(resp, "usage", None)
        if usage is not None:
            print(f"response_usage={usage}")
        used_model = getattr(resp, "model", None)
        if used_model is not None:
            print(f"response_model={used_model}")
        if hasattr(resp, "model_dump"):
            try:
                print("response_dump=" + json.dumps(resp.model_dump(), ensure_ascii=True, indent=2))
            except Exception:
                pass
        output_text = getattr(resp, "output_text", None)
        if output_text is not None:
            print(output_text)
        else:
            print(resp)

    asyncio.run(_run())
