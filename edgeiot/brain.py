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

from edgeiot.config import OllamaConfig

logger = logging.getLogger(__name__)

# ── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt(sensor: dict, style: str = "hybrid") -> str:
    """Create a natural-language prompt for the LLM from a sensor dict.

    Styles
    ------
    raw     : dumps the JSON verbatim.
    compact : terse one-liner.
    hybrid  : friendly description + key-value list (default).
    """
    if style == "raw":
        return (
            "You are the brain of a smart home running on an edge device.\n"
            f"Sensor reading: {json.dumps(sensor)}\n"
            "What action should the home take? Reply with a short answer."
        )

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

# Keywords for the seven canonical actions (edgeiot.semantic.actions)
_ACTION_KEYWORDS = {
    "security_alert":   ["security alert", "intruder", "call security",
                         "alarm"],
    "adjust_ac":        ["turn on air conditioner", "turn on the ac",
                         "cool the room", "lower the temperature",
                         "activate cooling", "turn on cooling",
                         "air conditioning", "heating", "raise the temperature"],
    "dehumidifier_on":  ["dehumidifier", "reduce humidity", "lower the humidity"],
    "ventilation_on":   ["ventilation", "open windows", "open the windows",
                         "turn on the fan"],
    "lights_off":       ["turn off lights", "turn off the lights",
                         "switch off lights"],
    "lights_on":        ["turn on lights", "turn on the lights",
                         "switch on lights"],
    "no_action":        ["no action", "no immediate action",
                         "everything is normal", "no changes needed"],
}


def parse_reply(reply: str) -> dict:
    """Extract a structured action from the LLM plain-text reply."""
    lowered = reply.lower()
    for action, keywords in _ACTION_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return {"action": action}
    return {"action": "other", "note": "no recognised keyword matched"}


# ── Public entry point ───────────────────────────────────────────────────────

def run(
    sensor: dict,
    config: OllamaConfig | None = None,
    style: str = "hybrid",
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

    prompt = build_prompt(sensor, style)

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
