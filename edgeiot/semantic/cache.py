"""
cache.py — Response cache for LLM decisions over the semantic vocabulary.

Semantic tokenization turns continuous readings into a small, finite set of
inputs (state + events), so the same input recurs many times. The cache stores
the decision made for an input the first time it is seen and returns it for
every later occurrence, without calling the LLM again.

Each new entry is validated once, when it is stored:
  * inputs inside the closed vocabulary are checked against the rule baseline;
    if the LLM's action differs, the rule's action is stored instead
    (policy "rules") or the entry is only flagged for review (policy "flag");
  * inputs outside the vocabulary (unseen symbols, free text) cannot be checked
    by the rules and are stored as answered, flagged for review.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

from edgeiot.semantic.actions import rule_action

# Symbols the tokenizer can produce (closed vocabulary of the grammar)
VOCABULARY = {
    "[TEMP_HIGH]", "[TEMP_LOW]", "[HUMIDITY_HIGH]", "[HUMIDITY_LOW]", "[NOISE_SPIKE]",
    "[MOTION_ACTIVE]", "[LOW_LIGHT]", "[ANOMALY_ALERT]", "[NO_EVENT]",
}

Key = Tuple[str, Tuple[str, ...], str]


class ResponseCache:
    """Exact-match cache keyed on (state, events, free-text note)."""

    def __init__(self, policy: str = "rules"):
        if policy not in ("rules", "flag"):
            raise ValueError("policy must be 'rules' or 'flag'")
        self.policy = policy
        self.entries: Dict[Key, Dict] = {}
        self.hits = 0
        self.misses = 0
        self.corrected = 0
        self.flagged = 0

    @staticmethod
    def key(state: str, tokens: List[str], note: str = "") -> Key:
        return (state, tuple(tokens) if tokens else ("[NO_EVENT]",), note.strip())

    @staticmethod
    def in_vocabulary(tokens: List[str], note: str = "") -> bool:
        return not note and all(t in VOCABULARY for t in tokens)

    def lookup(self, state: str, tokens: List[str], note: str = "") -> Optional[Dict]:
        entry = self.entries.get(self.key(state, tokens, note))
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def store(self, state: str, tokens: List[str], action: Optional[str],
              reasoning: str = "", note: str = "") -> Dict:
        """Validate a new LLM decision once and store it."""
        entry = {"action": action, "reasoning": reasoning, "source": "llm", "review": False}
        if self.in_vocabulary(tokens, note):
            expected = rule_action(state, tokens)
            if action != expected:
                if self.policy == "rules":
                    entry.update(action=expected, source="rule-corrected", llm_action=action)
                    self.corrected += 1
                else:
                    entry["review"] = True
                    self.flagged += 1
        else:
            entry["review"] = True
            self.flagged += 1
        self.entries[self.key(state, tokens, note)] = entry
        return entry

    def decide(self, state: str, tokens: List[str], ask_llm: Callable[[], Tuple[Optional[str], str]],
               note: str = "") -> Dict:
        """Return the cached decision, or ask the LLM once and cache its validated answer."""
        entry = self.lookup(state, tokens, note)
        if entry is not None:
            return dict(entry, cached=True)
        action, reasoning = ask_llm()
        return dict(self.store(state, tokens, action, reasoning, note), cached=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> Dict:
        return {"entries": len(self.entries), "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hit_rate, "corrected": self.corrected, "flagged": self.flagged}


def time_lookup(cache: ResponseCache, keys: List[Tuple[str, List[str]]], repeat: int = 10) -> float:
    """Mean time of a cache lookup in microseconds (does not change the hit counters)."""
    start = time.perf_counter()
    for _ in range(repeat):
        for state, tokens in keys:
            cache.entries.get(cache.key(state, tokens))
    return (time.perf_counter() - start) / (repeat * len(keys)) * 1e6
