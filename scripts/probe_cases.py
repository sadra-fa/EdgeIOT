"""
probe_cases.py — Hand-written cases for comparing the LLM with the rule baseline.

The acceptable actions were fixed before any model was run. Three groups:
  policy   : tokens from the closed vocabulary, several of them in conflict;
             the acceptable set follows the stated priority (safety > comfort
             > lighting), which the rule baseline implements by construction.
  unseen   : tokens that are NOT in the tokenizer's vocabulary (e.g. a smoke
             detector added later); the rule baseline has no branch for them.
  freetext : a resident's sentence in addition to the tokens.
"""

PROBE_CASES = [
    # id, group, state, tokens, note, acceptable actions
    ("P1", "policy", "sleeping", ["[TEMP_HIGH]", "[ANOMALY_ALERT]"], "", {"security_alert"}),
    ("P2", "policy", "sleeping", ["[MOTION_ACTIVE]", "[NOISE_SPIKE]", "[ANOMALY_ALERT]"], "", {"security_alert"}),
    ("P3", "policy", "active", ["[TEMP_HIGH]", "[HUMIDITY_HIGH]", "[ANOMALY_ALERT]"], "", {"security_alert"}),
    ("P4", "policy", "sedentary", ["[TEMP_HIGH]", "[HUMIDITY_HIGH]"], "", {"adjust_ac", "dehumidifier_on"}),
    ("P5", "policy", "sedentary", ["[LOW_LIGHT]"], "", {"lights_on"}),
    ("P6", "policy", "sedentary", ["[TEMP_LOW]", "[LOW_LIGHT]"], "", {"adjust_ac", "lights_on"}),
    ("P7", "policy", "sedentary", [], "", {"no_action"}),
    ("P8", "policy", "sleeping", ["[HUMIDITY_HIGH]", "[NOISE_SPIKE]"], "", {"dehumidifier_on"}),

    ("U1", "unseen", "sleeping", ["[SMOKE_DETECTED]"], "", {"security_alert", "ventilation_on"}),
    ("U2", "unseen", "sleeping", ["[WATER_LEAK]"], "", {"security_alert"}),
    ("U3", "unseen", "sleeping", ["[DOOR_OPEN]"], "", {"security_alert"}),
    ("U4", "unseen", "sedentary", ["[CO2_HIGH]"], "", {"ventilation_on"}),
    ("U5", "unseen", "sedentary", ["[AIR_STALE]"], "", {"ventilation_on"}),
    ("U6", "unseen", "sleeping", ["[GLASS_BREAK]"], "", {"security_alert"}),
    ("U7", "unseen", "sedentary", ["[MOLD_RISK]"], "", {"dehumidifier_on", "ventilation_on"}),
    ("U8", "unseen", "active", ["[GAS_LEAK]"], "", {"security_alert", "ventilation_on"}),

    ("N1", "freetext", "sedentary", [], "It is getting too dark to read in here.", {"lights_on"}),
    ("N2", "freetext", "sedentary", [], "I am going to bed now, good night.", {"lights_off"}),
    ("N3", "freetext", "sedentary", [], "The kitchen smells of burnt toast.", {"ventilation_on"}),
    ("N4", "freetext", "sedentary", [], "It feels really muggy today.", {"dehumidifier_on", "adjust_ac"}),
    ("N5", "freetext", "sedentary", [], "I'm freezing.", {"adjust_ac"}),
    ("N6", "freetext", "sleeping", [], "Someone is trying to open the back door!", {"security_alert"}),
]
