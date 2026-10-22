"""
actions.py — Canonical action taxonomy, JSON schema check and rule baseline.

The LLM is asked to answer with a JSON object of the form

    {"reasoning": "<one sentence>", "action_json": {"action": "<action>", ...}}

where <action> is one of CANONICAL_ACTIONS. `check_reply` measures how well a
reply follows that contract (the Action Syntactic Validity Rate), and
`rule_action` is the deterministic if-else controller the LLM is compared with.
"""

import json
import re
from typing import Dict, List, Optional

CANONICAL_ACTIONS = (
    "adjust_ac",
    "lights_off",
    "lights_on",
    "security_alert",
    "ventilation_on",
    "dehumidifier_on",
    "no_action",
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _first_json_object(text: str) -> Optional[dict]:
    """Return the first decodable JSON object embedded anywhere in `text`."""
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"\{", text)):
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _extract_action(obj: dict) -> Optional[str]:
    action_json = obj.get("action_json")
    if isinstance(action_json, dict):
        action = action_json.get("action")
    elif isinstance(action_json, str):
        action = action_json
    else:
        action = obj.get("action")
    return action.strip().lower() if isinstance(action, str) else None


def extract_answer(reply: str) -> Optional[dict]:
    """Return {"reasoning", "action_json"} recovered from a reply, or None.

    Accepts a strict JSON reply as well as a JSON object inside a code fence
    or after a preamble, provided it has a string reasoning and a canonical action.
    """
    text = reply.strip()
    fence = _FENCE.search(text)
    obj = _first_json_object(fence.group(1) if fence else text)
    if obj is None or not isinstance(obj.get("reasoning"), str):
        return None
    action = _extract_action(obj)
    if action not in CANONICAL_ACTIONS:
        return None
    action_json = obj["action_json"] if isinstance(obj.get("action_json"), dict) else {}
    return {"reasoning": obj["reasoning"], "action_json": dict(action_json, action=action)}


def check_reply(reply: str) -> Dict:
    """Classify a reply against the output contract.

    json_strict  : the whole reply (whitespace stripped) is one JSON object
    json_lenient : a JSON object can be recovered (code fence or preamble)
    schema_valid : json_strict, a string "reasoning", an "action_json" object
                   and an action from CANONICAL_ACTIONS  (counts towards R_valid)
    action       : the recovered action (from the lenient parse), or None
    """
    text = reply.strip()
    strict_obj = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            strict_obj = parsed
    except json.JSONDecodeError:
        pass

    lenient_obj = strict_obj
    if lenient_obj is None:
        fence = _FENCE.search(text)
        lenient_obj = _first_json_object(fence.group(1) if fence else text)

    action = _extract_action(lenient_obj) if lenient_obj else None
    schema_valid = bool(
        strict_obj is not None
        and isinstance(strict_obj.get("reasoning"), str)
        and isinstance(strict_obj.get("action_json"), dict)
        and _extract_action(strict_obj) in CANONICAL_ACTIONS
    )
    return {
        "json_strict": strict_obj is not None,
        "json_lenient": lenient_obj is not None,
        "schema_valid": schema_valid,
        "action": action,
        "action_canonical": action in CANONICAL_ACTIONS,
    }


def rule_action(state: str, tokens: List[str]) -> str:
    """Deterministic if-else controller over the closed token vocabulary.

    Priority: safety first, then comfort, then lighting.
    """
    t = set(tokens)
    if "[ANOMALY_ALERT]" in t:
        return "security_alert"
    if "[TEMP_HIGH]" in t or "[TEMP_LOW]" in t:
        return "adjust_ac"
    if "[HUMIDITY_HIGH]" in t:
        return "dehumidifier_on"
    if "[LOW_LIGHT]" in t and state != "sleeping":
        return "lights_on"
    if state == "sleeping" and "[LOW_LIGHT]" not in t:
        return "lights_off"
    return "no_action"
