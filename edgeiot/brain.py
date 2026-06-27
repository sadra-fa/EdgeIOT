"""
brain.py – Smart-home "brain" module.

Accepts a sensor-data dict, builds a natural-language prompt, queries the
local LLaMA 3.2 (3 B) model through the Ollama REST API, and returns:
  • the raw plain-text LLM reply
  • a JSON-structured action parsed from the reply
  • the round-trip latency in milliseconds

This module reuses the same Ollama HTTP endpoint as llm_reasoner.py so
that there is no dependency on the Ollama CLI binary being on $PATH.
"""

import json
import logging
import time
from typing import Dict

import requests
from typing import List, Dict, Any

from edgeiot.config import OllamaConfig

logger = logging.getLogger(__name__)

# ── Helper ─────────────────────────────────────────────────────────────────────
def format_history(history: List[Dict[str, Any]]) -> str:
    """Return a markdown table summarising recent events.
    Each entry should be a dict with keys:
        - "sensor": the sensor dict from that cycle
        - "action": the parsed action dict from that cycle
    Only the last 1 event is shown (as per user request).
    """
    if not history:
        return ""
    lines = ["| Timestamp | Temp (°C) | Action |", "|---|---|---|"]
    for entry in history[-1:]:
        sensor = entry.get("sensor", {})
        action = entry.get("action", {}).get("action", "")
        ts = sensor.get("timestamp", "")
        temp = sensor.get("temperature", "")
        lines.append(f"| {ts} | {temp} | {action} |")
    return "\n".join(lines) + "\n"



# ── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt(sensor: dict, desired_temp: float | None = None, history: list[dict] | None = None, style: str = "hybrid") -> str:
    """Create a natural-language prompt for the LLM from a sensor dict.

    Styles
    ------
    raw     : dumps the JSON verbatim.
    compact : terse one-liner.
    hybrid  : friendly description + key-value list (default).
    """
    # Base description
    base_prompt = (
        "You are an AI assistant embedded in a smart‑home edge device.\n"
        "The sensors currently report:\n"
        f"  • Temperature : {sensor['temperature']} °C\n"
        f"  • Motion detected : {'yes' if sensor['motion'] else 'no'}\n"
        f"  • Noise level : {sensor['noise']} dB\n"
    )

    # Desired temperature context
    if desired_temp is not None:
        base_prompt += f"Desired temperature is set to {desired_temp} °C (acceptable range 15‑25 °C).\nIf the current temperature is below the desired value, the system should turn on the heater; if above, it should turn on the air conditioner.\n"

    # History context (last few events)
    if history:
        base_prompt += "Recent history:\n"
        base_prompt += format_history(history)

    # Action request
    base_prompt += "Based on this information, decide what action the smart home should take. Reply in one concise paragraph."
    return base_prompt

    if style == "compact":
        motion_str = "motion detected" if sensor["motion"] else "no motion"
        return (
            f"Smart-home status — temp {sensor['temperature']}°C, "
            f"{motion_str}, noise {sensor['noise']} dB. "
            "What action should the home take?"
        )

    # hybrid (default)
    motion_str = "yes" if sensor["motion"] else "no"
    return (
        "You are an AI assistant embedded in a smart-home edge device.\n"
        "The sensors currently report:\n"
        f"  • Temperature : {sensor['temperature']} °C\n"
        f"  • Motion detected : {motion_str}\n"
        f"  • Noise level : {sensor['noise']} dB\n\n"
        "Based on this information, decide what action the smart home "
        "should take.  Reply in one concise paragraph."
    )


# ── Ollama caller ────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, config: OllamaConfig) -> Dict:
    """Send a prompt to the Ollama REST API and return the raw response dict."""
    payload = {
        "model":  config.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 150,
            "num_ctx":     512,
        },
    }
    resp = requests.post(config.url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


# ── Reply parser ─────────────────────────────────────────────────────────────

_ACTION_KEYWORDS = {
    "adjust_ac":        ["turn on air conditioner", "turn on the ac",
                         "cool the room", "lower the temperature",
                         "activate cooling", "turn on cooling",
                         "air conditioning", "decrease temperature", "lower the temperature", "cool down"],
    "turn_off_ac":      ["turn off air conditioner", "turn off the ac",
                         "disable cooling"],
    "turn_on_ac":       ["turn on air conditioner", "activate cooling", "start AC", "cool down", "lower the temperature", "decrease temperature"],
    "turn_off_ac":      ["turn off air conditioner", "disable cooling", "stop AC"],
    "turn_on_heater":   ["turn on heater", "turn on the heater", "activate heating", "start heating", "increase temperature", "raise temperature", "heat up"],
    "turn_off_heater":  ["turn off heater", "disable heating", "stop heating", "stop heating", "turn off heating"],
    "close_windows":    ["close windows", "close the windows",
                         "shut the windows"],
    "open_windows":     ["open windows", "open the windows"],
    "turn_off_lights":  ["turn off lights", "turn off the lights",
                         "switch off lights"],
    "turn_on_lights":   ["turn on lights", "turn on the lights",
                         "switch on lights"],
    "reduce_noise":     ["reduce noise", "lower the volume",
                         "noise reduction"],
    "security_alert":   ["security alert", "intruder", "call security",
                         "alarm"],
    "no_action":        ["no action", "no immediate action",
                         "everything is normal", "no changes needed"],
}


def parse_reply(reply: str) -> dict:
    """Extract a structured action from the LLM plain-text reply.
    The function first checks the explicit keyword lists. If none match,
    it falls back to generic matching for common HVAC terms.
    """
    lowered = reply.lower()
    for action, keywords in _ACTION_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return {"action": action}
    # Generic fallback checks – these catch truncated or paraphrased phrasing.
    if "heater" in lowered or "heating" in lowered:
        return {"action": "turn_on_heater"}
    if "air conditioner" in lowered or "air conditioning" in lowered or "ac" in lowered:
        return {"action": "adjust_ac"}
    return {"action": "other", "note": "no recognised keyword matched"}


# ── Public entry point ───────────────────────────────────────────────────────

def run(
    sensor: dict,
    config: OllamaConfig | None = None,
    style: str = "hybrid",
    desired_temp: float | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Full cycle: build prompt → query LLM → parse reply.

    Parameters
    ----------
    sensor : dict   – keys ``temperature``, ``motion``, ``noise``.
    config : OllamaConfig – connection details (defaults to localhost).
    style  : str    – prompt style: ``raw``, ``compact``, or ``hybrid``.

    Returns
    -------
    dict with keys:
        sensor, prompt, raw_reply, action, latency_ms
    """
    if config is None:
        config = OllamaConfig()

    prompt = build_prompt(sensor, desired_temp=desired_temp, history=history, style=style)

    start = time.perf_counter()
    try:
        data = _call_ollama(prompt, config)
        raw_reply = data.get("response", "").strip()
    except Exception as exc:
        logger.error("Ollama call failed: %s", exc)
        raw_reply = f"[LLM ERROR: {exc}]"
    latency_ms = (time.perf_counter() - start) * 1000

    action = parse_reply(raw_reply)

    return {
        "sensor":     sensor,
        "prompt":     prompt,
        "raw_reply":  raw_reply,
        "action":     action,
        "latency_ms": round(latency_ms, 2),
    }
