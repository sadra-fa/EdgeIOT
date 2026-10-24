#!/usr/bin/env python3
"""
analyze_results.py — Turn the raw experiment CSVs into the dissertation's
numbers, tables and figures.

Reads the latest pipeline run (outputs/rerun/*) and the latest directory of
each experiment (outputs/experiments/<experiment>_*), and writes
  dissertation/latex/results_macros.tex   \\newcommand per reported number
  dissertation/latex/tables/*.tex         result tables
  dissertation/latex/figures/results/*.pdf
so that every number in Chapter 5 is computed from the stored measurements.

Usage:
    python scripts/analyze_results.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
TEX = ROOT / "dissertation" / "latex"
TABLES = TEX / "tables"
FIGS = TEX / "figures" / "results"
NUM_PREDICT = 128

MODEL_KEYS = {
    "llama3.2:3b": ("Llama", "LLaMA 3.2 (3B)"),
    "phi3.5:latest": ("Phi", "Phi-3.5 (3.8B)"),
    "qwen2.5:1.5b": ("Qwen", "Qwen 2.5 (1.5B)"),
    "edgeiot-llama3.2-ft-adapter": ("FtAdapter", "LLaMA 3.2 FT (adapter)"),
    "edgeiot-llama3.2-ft-merged": ("FtMerged", "LLaMA 3.2 FT (merged)"),
    "edgeiot-phi3.5-ft-merged": ("PhiFtMerged", "Phi-3.5 FT (merged)"),
    "edgeiot-qwen2.5-ft-merged": ("QwenFtMerged", "Qwen 2.5 FT (merged)"),
    "edgeiot-llama3.2-ft-fp16merge": ("FtFpMerge", "LLaMA 3.2 FT (FP16 merge)"),
    "edgeiot-qwen2.5-ft-fp16merge": ("QwenFtFpMerge", "Qwen 2.5 FT (FP16 merge)"),
    "edgeiot-llama3.2-ft-q8": ("FtQeight", "LLaMA 3.2 FT (Q8_0)"),
    "edgeiot-qwen2.5-ft-q8": ("QwenFtQeight", "Qwen 2.5 FT (Q8_0)"),
    "edgeiot-phi3.5-ft-q8": ("PhiFtQeight", "Phi-3.5 FT (Q8_0)"),
    "edgeiot-phi3.5-ft-q40": ("PhiFtQfourZero", "Phi-3.5 FT (Q4_0)"),
    "edgeiot-qwen2.5-ft-f16": ("QwenFtFsixteen", "Qwen 2.5 FT (FP16 GGUF)"),
}
TOKENIZERS = {
    "llama3.2:3b": ROOT / "languagemodels" / "Llama-3.2-3B-Instruct" / "tokenizer.json",
    "qwen2.5:1.5b": ROOT / "languagemodels" / "Qwen2.5-1.5B-Instruct-tokenizer" / "tokenizer.json",
    "phi3.5:latest": ROOT / "languagemodels" / "Phi-3.5-mini-instruct-tokenizer" / "tokenizer.json",
}

# Palette (validated: reference categorical slots 1-3 on white) + neutral
C_PREFILL, C_DECODE, C_OTHER = "#2a78d6", "#eb6834", "#b4b2a9"
INK, INK2 = "#0b0b0b", "#52514e"

macros = {}


def macro(name, value):
    macros[name] = value


def latest(pattern):
    """Latest directory matching `pattern`; experiment directories count only when finished."""
    dirs = [d for d in sorted(OUT.glob(pattern))
            if "experiments" not in pattern or (d / "requests.csv").exists()]
    return dirs[-1] if dirs else None


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:,.{nd}f}"


def pct(x, nd=0):
    return fmt(100 * x, nd) + "\\,\\%"


def p_value(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return float("nan")
    return wilcoxon(a, b).pvalue


def fmt_p(p):
    if np.isnan(p):
        return "--"
    return "\\ensuremath{<0.001}" if p < 0.001 else f"{p:.3f}"


def latex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in s)


def write_table(name, body):
    (TABLES / f"{name}.tex").write_text(body)


def prepare(df):
    if "warmup" in df.columns:
        df = df[df["warmup"] != True]  # noqa: E712
    df = df.copy()
    df["other_ms"] = df["wall_ms"] - df["prefill_ms"] - df["decode_ms"] - df["load_ms"]
    df["capped"] = df["eval_count"] >= NUM_PREDICT
    df["energy_j"] = df["power_mean"] / 1000 * df["wall_ms"] / 1000
    return df


def summarise(g):
    recov = g["action_canonical"].astype(bool)
    return {
        "n": len(g),
        "np": g["prompt_eval_count"].mean(),
        "prefill_med": g["prefill_ms"].median(),
        "no": g["eval_count"].mean(),
        "decode_med": g["decode_ms"].median(),
        "other_med": g["other_ms"].median(),
        "total_med": g["wall_ms"].median(),
        "total_mean": g["wall_ms"].mean(),
        "total_p95": g["wall_ms"].quantile(0.95),
        "total_sd": g["wall_ms"].std(),
        "tps": g["tokens_per_sec"].mean(),
        "tps_sd": g["tokens_per_sec"].std(),
        "prefill_tps": (g["prompt_eval_count"] / (g["prefill_ms"] / 1000)).median(),
        "gpu": g["gpu_mean"].mean(),
        "ram": g["ram_mean"].mean(),
        "temp_max": g["temp_max"].max(),
        "power": g["power_mean"].mean(),
        "energy": g["energy_j"].mean(),
        "asvr": g["schema_valid"].mean(),
        "recoverable": recov.mean(),
        "capped": g["capped"].mean(),
        "over80": (g["eval_count"] > 80).mean(),
        "asvr_uncapped": g.loc[~g["capped"], "schema_valid"].mean() if (~g["capped"]).any() else np.nan,
        "agree": (g.loc[recov, "action"] == g.loc[recov, "rule_action"]).mean() if "rule_action" in g and recov.any() else np.nan,
        "fenced": g["response"].fillna("").str.contains("```").mean(),
        "after_fence": g["response"].fillna("").apply(
            lambda r: len(r.split("```")) >= 3 and r.split("```")[2].strip() != "").mean(),
        "invented": (g["action"].notna() & ~g["action_canonical"].astype(bool)).mean(),
    }


def stacked_latency_figure(rows, fname, title=None):
    """Horizontal stacked bars of median prefill / decode / other per row.

    Each row is (label, prefill, decode, other, total); the bar end is labelled
    with the median end-to-end latency, which need not equal the sum of medians."""
    labels = [r[0] for r in rows]
    pre = np.array([r[1] for r in rows]) / 1000
    dec = np.array([r[2] for r in rows]) / 1000
    oth = np.clip(np.array([r[3] for r in rows]) / 1000, 0, None)
    fig, ax = plt.subplots(figsize=(6.3, 0.48 * len(rows) + 1.1))
    y = np.arange(len(rows))[::-1]
    kw = dict(height=0.56, edgecolor="white", linewidth=1.2)
    ax.barh(y, pre, color=C_PREFILL, label="Prefill", **kw)
    ax.barh(y, dec, left=pre, color=C_DECODE, label="Decode", **kw)
    ax.barh(y, oth, left=pre + dec, color=C_OTHER, label="Other", **kw)
    totals = np.array([r[4] for r in rows]) / 1000
    for yi, end, tot in zip(y, pre + dec + oth, totals):
        ax.text(end + 0.05, yi, f"{tot:.2f} s", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([re.sub(r"\$.*?\$", "", l) for l in labels], fontsize=8.5, color=INK)
    ax.set_xlabel("Median latency per request (s)", fontsize=8.5, color=INK2)
    ax.set_xlim(0, (pre + dec + oth).max() * 1.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d6d0")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.grid(True, color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(FIGS / fname)
    plt.close(fig)


# ── E0: pipeline ─────────────────────────────────────────────────────────────
def pipeline():
    run = latest("rerun/*")
    m = json.loads((run / "pipeline_metrics.json").read_text())
    acc = m["accuracy"]
    cv = m["cv"]
    macro("resAccDT", fmt(acc["Decision Tree"], 4))
    macro("resAccRF", fmt(acc["Random Forest"], 4))
    macro("resAccXGB", fmt(acc["XGBoost"], 4))
    macro("resPcaVar", fmt(100 * sum(m["pca_explained_variance"]), 1))
    macro("resPcaVarOne", fmt(100 * m["pca_explained_variance"][0], 1))
    macro("resPcaVarTwo", fmt(100 * m["pca_explained_variance"][1], 1))
    macro("resAnomWindows", fmt(m["anomaly_windows"]))
    macro("resAnomFlagged", fmt(m["anomaly_flagged"]))
    macro("resAnomThreshold", fmt(-m["anomaly_offset"], 3))
    macro("resAnomTest", fmt(m["test_rows_with_anomaly_flag"]))
    macro("resAnomTestPct", fmt(100 * m["test_rows_with_anomaly_flag"] / 6000, 1))

    rows = [f"{name} & {fmt(acc[name], 4)} & {fmt(cv[name][0], 4)} $\\pm$ {fmt(cv[name][1], 4)} \\\\"
            for name in ("Decision Tree", "Random Forest", "XGBoost")]
    write_table("classification", "\n".join(rows) + "\n")

    sizes = m["cluster_sizes"]
    names = {"0": "sedentary", "1": "active", "2": "sleeping"}
    beh = m["cluster_vs_behaviour"]
    means = m["cluster_sensor_means"]
    lines = []
    for c in ("0", "1", "2"):
        comp = ", ".join(f"{b} {beh[b][c]:,}" for b in beh if beh[b].get(c, 0) > 0)
        lines.append(
            f"{c} & \\texttt{{{names[c]}}} & {int(sizes[c]):,} & {comp} & "
            f"{means['temperature'][c]:.1f} & {means['noise_db'][c]:.1f} & "
            f"{means['light_intensity'][c]:.0f} & {means['motion_level'][c]:.2f} \\\\")
    write_table("clusters", "\n".join(lines) + "\n")

    leak = run / "leakage_check.json"
    if leak.exists():
        lk = json.loads(leak.read_text())
        macro("resLeakAll", fmt(lk["all_24_features"]["accuracy"], 4))
        macro("resLeakLoc", fmt(lk["sensors_and_location"]["accuracy"], 4))
        macro("resLeakSensors", fmt(lk["sensors_only"]["accuracy"], 4))

    # Is the row order a time series? Lag-1 autocorrelation and location changes
    raw = pd.read_csv(ROOT / "RealLife.csv")
    sensors = ["temperature", "humidity", "noise_db", "light_intensity", "motion_level"]
    ac = [raw[c].fillna(raw[c].mean()).autocorr(1) for c in sensors]
    macro("resAutocorrMax", fmt(max(abs(a) for a in ac), 3))
    macro("resLocChange", fmt(100 * (raw["location"] != raw["location"].shift()).iloc[1:].mean(), 1))
    p = raw["location"].value_counts(normalize=True)
    macro("resLocChangeShuffled", fmt(100 * (1 - (p ** 2).sum()), 1))

    inputs = pd.read_csv(run / "llm_inputs.csv")
    inputs["tokens"] = inputs["tokens"].fillna("")
    macro("resDistinctInputs", fmt((inputs["state"] + inputs["tokens"]).nunique()))
    macro("resDistinctEvents", fmt(inputs["tokens"].nunique()))
    macro("resNoEventPct", fmt(100 * (inputs["tokens"] == "").mean(), 1))

    # Response cache replay (scripts/cache_replay.py)
    cr = run / "cache_replay.json"
    if cr.exists():
        c = json.loads(cr.read_text())
        r = c["rules"]
        macro("resCacheRows", fmt(c["rows"]))
        macro("resCacheCalls", fmt(r["llm_calls"]))
        macro("resCacheHitRate", fmt(100 * r["hit_rate"], 1))
        macro("resCacheCorrected", fmt(r["corrected"]))
        macro("resCacheCorrectedPct", fmt(100 * r["corrected"] / r["entries"], 0))
        macro("resCacheLookupUs", fmt(c["lookup_us"], 2))
        macro("resCacheRawHitRate", fmt(100 * c["raw_hit_rate"], 1))
        macro("resCacheNotLogged", fmt(r["answers_not_logged"]))
        curve = pd.read_csv(run / "cache_curve.csv")
        for n, name in ((100, "Hundred"), (500, "FiveHundred"), (1000, "Thousand")):
            macro(f"resCacheHit{name}", fmt(100 * curve.loc[curve["rows"] == n, "hit_rate"].iloc[0], 0))
        rows = [f"{int(x.rows):,} & {int(x.llm_calls):,} & {100 * x.hit_rate:.1f}\\,\\% \\\\"
                for x in curve.itertuples()]
        write_table("cache_curve", "\n".join(rows) + "\n")

    # Cost of the rule baseline over the same inputs
    import sys, time
    sys.path.insert(0, str(ROOT))
    from edgeiot.semantic.actions import rule_action
    pairs = list(zip(inputs["state"], inputs["tokens"].str.split()))
    start = time.perf_counter()
    for _ in range(10):
        for state, toks in pairs:
            rule_action(state, toks)
    per_call_us = (time.perf_counter() - start) / (10 * len(pairs)) * 1e6
    macro("resRuleMicros", fmt(per_call_us, 1))
    return inputs


# ── E1: multi-model ──────────────────────────────────────────────────────────
def multi():
    d = latest("experiments/multi_*")
    if d is None:
        return None
    df = prepare(pd.read_csv(d / "requests.csv"))
    info = pd.read_csv(d / "models.csv").set_index("model")
    meta = json.loads((d / "meta.json").read_text())
    macro("resIdleGpu", fmt(meta["idle"]["gpu_mean"], 0))
    macro("resIdleRam", fmt(meta["idle"]["ram_mean"], 0))
    macro("resIdlePower", fmt(meta["idle"]["power_mean"] / 1000, 1))
    macro("resSampleN", fmt(df.groupby("model").size().iloc[0]))
    macro("resSampleAnom", fmt((df[df["model"] == df["model"].iloc[0]]["anomaly"] == "ABNORMAL").sum()))

    perf, val, fig_rows, summ = [], [], [], {}
    for model in [m for m in MODEL_KEYS if m in set(df["model"])]:
        key, label = MODEL_KEYS[model]
        s = summarise(df[df["model"] == model])
        summ[model] = s
        mi = info.loc[model]
        names = {"total_med": "TotalMed", "total_p95": "TotalPctl", "total_mean": "TotalMean",
                 "prefill_med": "PrefillMed", "decode_med": "DecodeMed", "np": "Np", "no": "No",
                 "tps": "Tps", "gpu": "Gpu", "energy": "Energy", "power": "Power"}
        for k, name in names.items():
            value = s[k] / 1000 if k == "power" else s[k]
            macro(f"res{key}{name}", fmt(value, 1 if k in ("tps", "no", "np", "gpu", "power") else 0))
        macro(f"res{key}Asvr", fmt(100 * s["asvr"], 0))
        macro(f"res{key}Recoverable", fmt(100 * s["recoverable"], 0))
        macro(f"res{key}Capped", fmt(100 * s["capped"], 0))
        macro(f"res{key}OverEighty", fmt(100 * s["over80"], 0))
        macro(f"res{key}Agree", fmt(100 * s["agree"], 0))
        macro(f"res{key}AfterFence", fmt(100 * s["after_fence"], 0))
        macro(f"res{key}Invented", fmt(100 * s["invented"], 0))
        macro(f"res{key}Mem", fmt(mi["ps_size_mb"], 0))
        macro(f"res{key}Cold", fmt(mi["cold_load_ms"] / 1000, 1))
        macro(f"res{key}PrefillTps", fmt(s["prefill_tps"], 0))
        perf.append(
            f"{label} & {fmt(s['np'], 0)} & {fmt(s['prefill_med'], 0)} & {fmt(s['no'], 1)} & "
            f"{fmt(s['decode_med'], 0)} & {fmt(s['total_med'], 0)} & {fmt(s['total_p95'], 0)} & "
            f"{fmt(s['tps'], 1)} \\\\")
        val.append(
            f"{label} & {fmt(mi['ps_size_mb'], 0)} & {fmt(mi['cold_load_ms'] / 1000, 1)} & "
            f"{fmt(s['gpu'], 0)} & {fmt(s['ram'], 0)} & {fmt(s['power'] / 1000, 1)} & "
            f"{fmt(s['energy'], 1)} & {fmt(s['temp_max'], 1)} \\\\")
        fig_rows.append((label, s["prefill_med"], s["decode_med"], s["other_med"], s["total_med"]))
    # LLaMA's validity by state (code-writing failure mode)
    l = df[df["model"] == "llama3.2:3b"].copy()
    if not l.empty:
        l["code"] = l["response"].fillna("").str.contains(r"^\s*(?:import |from |class |def |```python)", regex=True)
        act, rest = l[l["state"] == "active"], l[l["state"] != "active"]
        macro("resLlamaValidActive", fmt(100 * act["schema_valid"].mean(), 0))
        macro("resLlamaValidOther", fmt(100 * rest["schema_valid"].mean(), 0))
        macro("resLlamaCodeActive", fmt(100 * act["code"].mean(), 0))
        macro("resLlamaActiveN", fmt(len(act)))
    write_table("multi_latency", "\n".join(perf) + "\n")
    write_table("multi_resources", "\n".join(val) + "\n")

    vrows = []
    for model, s in summ.items():
        vrows.append(
            f"{MODEL_KEYS[model][1]} & {pct(s['asvr'])} & {pct(s['recoverable'])} & "
            f"{pct(s['fenced'])} & {pct(s['capped'])} & {pct(s['over80'])} & {pct(s['agree'])} \\\\")
    write_table("multi_validity", "\n".join(vrows) + "\n")
    stacked_latency_figure(fig_rows, "multi_latency.pdf",
                           "Where the time goes: median prefill, decode and other time per request")
    df.to_csv(FIGS / "multi_requests_clean.csv", index=False)
    run = latest("rerun/*")
    cr = run / "cache_replay.json"
    if cr.exists() and "llama3.2:3b" in summ:
        c = json.loads(cr.read_text())
        calls, rows = c["rules"]["llm_calls"], c["rows"]
        mean_ms = (calls * summ["llama3.2:3b"]["total_mean"] + (rows - calls) * c["lookup_us"] / 1000) / rows
        macro("resCacheMeanMs", fmt(mean_ms, 0))
        macro("resCacheSpeedup", fmt(summ["llama3.2:3b"]["total_mean"] / mean_ms, 0))
    macro("resMaxPromptTokens", fmt(df["prompt_eval_count"].max()))
    return df, summ


# ── E2: tokenization ─────────────────────────────────────────────────────────
def tokenization():
    d = latest("experiments/tokenization_*")
    if d is None:
        return
    from tokenizers import Tokenizer
    df = prepare(pd.read_csv(d / "requests.csv"))
    rows, fig_rows, extra = [], [], []
    for model in [m for m in MODEL_KEYS if m in set(df["model"])]:
        key, label = MODEL_KEYS[model]
        tk = Tokenizer.from_file(str(TOKENIZERS[model])) if TOKENIZERS.get(model, Path()).exists() else None
        g = df[df["model"] == model].copy()
        if tk is not None:
            g["frag_tokens"] = [len(tk.encode(f, add_special_tokens=False).ids) for f in g["fragment"]]
        else:
            g["frag_tokens"] = np.nan
        sem = g[g["condition"] == "semantic"].sort_values("sample")
        raw = g[g["condition"] == "raw"].sort_values("sample")
        res = {}
        for cond, h in (("raw", raw), ("semantic", sem)):
            res[cond] = {
                "chars": h["fragment_chars"].mean(),
                "pred": h["n_predicates"].mean() if cond == "semantic" else np.nan,
                "frag_tok": h["frag_tokens"].mean(),
                "np": h["prompt_eval_count"].mean(),
                "prefill": h["prefill_ms"].median(),
                "no": h["eval_count"].mean(),
                "capped": h["capped"].mean(),
                "decode": h["decode_ms"].median(),
                "total": h["wall_ms"].median(),
                "total_mean": h["wall_ms"].mean(),
                "asvr": h["schema_valid"].mean(),
            }
            rows.append(
                f"{label if cond == 'raw' else ''} & {cond.capitalize()} & {fmt(res[cond]['chars'], 1)} & "
                f"{fmt(res[cond]['pred'], 2)} & {fmt(res[cond]['frag_tok'], 1)} & {fmt(res[cond]['np'], 1)} & "
                f"{fmt(res[cond]['prefill'], 0)} & {fmt(res[cond]['no'], 1)} & {pct(res[cond]['capped'])} & "
                f"{fmt(res[cond]['decode'], 0)} & {fmt(res[cond]['total'], 0)} \\\\")
            fig_rows.append((f"{label.split(' (')[0]} — {cond}", res[cond]["prefill"], res[cond]["decode"],
                             h["other_ms"].median(), res[cond]["total"]))
        r, s_ = res["raw"], res["semantic"]
        red = lambda a, b: 100 * (a - b) / a  # noqa: E731
        p_pre = p_value(raw["prefill_ms"], sem["prefill_ms"])
        p_dec = p_value(raw["decode_ms"], sem["decode_ms"])
        p_tot = p_value(raw["wall_ms"], sem["wall_ms"])
        p_out = p_value(raw["eval_count"], sem["eval_count"])
        rows.append(
            f" & \\textit{{Change}} & {fmt(-red(r['chars'], s_['chars']), 1)}\\,\\% & -- & "
            f"{fmt(-red(r['frag_tok'], s_['frag_tok']), 1)}\\,\\% & {fmt(-red(r['np'], s_['np']), 1)}\\,\\% & "
            f"{fmt(-red(r['prefill'], s_['prefill']), 1)}\\,\\% & {fmt(-red(r['no'], s_['no']), 1)}\\,\\% & -- & "
            f"{fmt(-red(r['decode'], s_['decode']), 1)}\\,\\% & {fmt(-red(r['total'], s_['total']), 1)}\\,\\% \\\\")
        if model != list(MODEL_KEYS)[-1]:
            rows.append("\\midrule")
        extra.append(f"{label} & {fmt_p(p_pre)} & {fmt_p(p_out)} & {fmt_p(p_dec)} & {fmt_p(p_tot)} \\\\")
        for name, val, nd in (
            ("CharsRaw", r["chars"], 1), ("CharsSem", s_["chars"], 1), ("Pred", s_["pred"], 2),
            ("FragTokRaw", r["frag_tok"], 1), ("FragTokSem", s_["frag_tok"], 1),
            ("FullTokRaw", r["np"], 1), ("FullTokSem", s_["np"], 1),
            ("PrefillRaw", r["prefill"], 0), ("PrefillSem", s_["prefill"], 0),
            ("OutRaw", r["no"], 1), ("OutSem", s_["no"], 1),
            ("DecodeRaw", r["decode"], 0), ("DecodeSem", s_["decode"], 0),
            ("TotalRaw", r["total"], 0), ("TotalSem", s_["total"], 0),
            ("CappedRaw", 100 * r["capped"], 0), ("CappedSem", 100 * s_["capped"], 0),
            ("AsvrRaw", 100 * r["asvr"], 0), ("AsvrSem", 100 * s_["asvr"], 0),
        ):
            macro(f"resTok{key}{name}", fmt(val, nd))
        macro(f"resTok{key}CharsRed", fmt(red(r["chars"], s_["chars"]), 1))
        macro(f"resTok{key}FragTokRed", fmt(red(r["frag_tok"], s_["frag_tok"]), 1))
        macro(f"resTok{key}FullTokRed", fmt(red(r["np"], s_["np"]), 1))
        macro(f"resTok{key}PrefillSave", fmt(r["prefill"] - s_["prefill"], 0))
        macro(f"resTok{key}PrefillRed", fmt(red(r["prefill"], s_["prefill"]), 1))
        macro(f"resTok{key}DecodeRed", fmt(red(r["decode"], s_["decode"]), 1))
        macro(f"resTok{key}TotalRed", fmt(red(r["total"], s_["total"]), 1))
        macro(f"resTok{key}OutRed", fmt(red(r["no"], s_["no"]), 1))
        macro(f"resTok{key}PTotal", fmt_p(p_tot))
        macro(f"resTok{key}PPrefill", fmt_p(p_pre))
        macro(f"resTok{key}PDecode", fmt_p(p_dec))
        if tk is not None:
            for tok in ("[HUMIDITY_HIGH]", "[TEMP_HIGH]", "[NO_EVENT]"):
                n = len(tk.encode(tok, add_special_tokens=False).ids)
                macro(f"resTok{key}Split{re.sub('[^A-Za-z]', '', tok.title())}", str(n))
    write_table("tokenization", "\n".join(rows) + "\n")
    write_table("tokenization_tests", "\n".join(extra) + "\n")
    stacked_latency_figure(fig_rows, "tokenization_latency.pdf",
                           "Raw vs. semantic fragment: median prefill, decode and other time")


# ── E3: probe ────────────────────────────────────────────────────────────────
def probe(pattern="experiments/probe_*", tag=""):
    dirs = [d for d in sorted(OUT.glob(pattern)) if (d / "requests.csv").exists()]
    if not dirs:
        return
    df = pd.concat([prepare(pd.read_csv(d / "requests.csv")) for d in dirs])
    df = df.drop_duplicates(subset=["model", "case"], keep="last")
    groups = ["policy", "unseen", "freetext"]
    rows = []
    rule = df[df["model"] == df["model"].iloc[0]]
    rule_acc = {g: rule.loc[rule["group"] == g, "rule_correct"].mean() for g in groups}
    rows.append("Rule baseline & " + " & ".join(pct(rule_acc[g]) for g in groups)
                + f" & {pct(rule['rule_correct'].mean())} \\\\")
    rows.append("\\midrule")
    for model in [m for m in MODEL_KEYS if m in set(df["model"])]:
        key, label = MODEL_KEYS[model]
        g = df[df["model"] == model]
        acc = {grp: g.loc[g["group"] == grp, "llm_correct"].mean() for grp in groups}
        rows.append(f"{label} & " + " & ".join(pct(acc[x]) for x in groups)
                    + f" & {pct(g['llm_correct'].mean())} \\\\")
        for grp in groups:
            macro(f"resProbe{key}{grp.title()}", fmt(100 * acc[grp], 0))
        macro(f"resProbe{key}All", fmt(100 * g["llm_correct"].mean(), 0))
    for grp in groups:
        macro(f"resProbeRule{grp.title()}", fmt(100 * rule_acc[grp], 0))
    models_present = set(df["model"])
    if {"llama3.2:3b", "edgeiot-llama3.2-ft-merged"} <= models_present:
        base_all = df[df["model"] == "llama3.2:3b"]["llm_correct"].mean()
        ft_all = df[df["model"] == "edgeiot-llama3.2-ft-merged"]["llm_correct"].mean()
        macro("resProbeFtDrop", fmt(100 * (base_all - ft_all), 0))
    write_table("probe", "\n".join(rows) + "\n")

    # Per-case matrix
    models = [m for m in MODEL_KEYS if m in set(df["model"])]
    lines = []
    short = {"policy": "P", "unseen": "U", "freetext": "N"}
    for cid in rule["case"]:
        r = rule[rule["case"] == cid].iloc[0]
        tokens = r["tokens"] if isinstance(r["tokens"], str) and r["tokens"] else "[NO_EVENT]"
        what = f"``{r['note']}''" if isinstance(r["note"], str) and r["note"] else tokens
        cells = []
        for model in models:
            row = df[(df["model"] == model) & (df["case"] == cid)]
            if row.empty:
                cells.append("")
                continue
            a = row.iloc[0]["action"]
            mark = "\\checkmark" if row.iloc[0]["llm_correct"] else "$\\times$"
            cells.append(mark)
        lines.append(f"{cid} & {r['state']} & {latex_escape(str(what))} & "
                     f"{' \\textbar{} '.join('\\texttt{' + latex_escape(x) + '}' for x in r['acceptable'].split('|'))} & "
                     f"{'\\checkmark' if r['rule_correct'] else '$\\times$'} & " + " & ".join(cells) + " \\\\")
    write_table("probe_cases", "\n".join(lines) + "\n")
    df.to_csv(FIGS / "probe_requests_clean.csv", index=False)


# ── E4: fine-tuning ──────────────────────────────────────────────────────────
# Base/fine-tuned pairs: (key, base tag, fine-tuned tag, experiment glob, corpus dir, adapter dir)
FT_PAIRS = [
    ("Llama", "llama3.2:3b", "edgeiot-llama3.2-ft-merged", "experiments/finetune_*",
     "data/finetune", "languagemodels/fine_tuned_llama3_lora"),
    ("Qwen", "qwen2.5:1.5b", "edgeiot-qwen2.5-ft-merged", "experiments/finetune-qwen_*",
     "data/finetune_qwen", "languagemodels/fine_tuned_qwen25_lora"),
    ("Phi", "phi3.5:latest", "edgeiot-phi3.5-ft-merged", "experiments/finetune-phi_*",
     "data/finetune_phi", "languagemodels/fine_tuned_phi35_lora"),
]


FT_STATS = {}   # (key, "base"/"ft") -> summary of the E4 run, filled by finetune()


def ft_prefixes(key):
    """Macro prefixes; LLaMA keeps the original names used throughout the text."""
    if key == "Llama":
        return dict(base="resFtBase", ft="resFtFtMerged", corpus="resCorpus", teacher="resTeacher", train="resTrain")
    return dict(base=f"resFt{key}Base", ft=f"resFt{key}Ft", corpus=f"resCorpus{key}",
                teacher=f"resTeacher{key}", train=f"resTrain{key}")


def finetune():
    rows, fig_rows, split_rows, frames = [], [], [], []
    for key, base_tag, ft_tag, pattern, corpus_dir, adapter_dir in FT_PAIRS:
        pre = ft_prefixes(key)
        corpus = ROOT / corpus_dir / "corpus_summary.json"
        if corpus.exists():
            c = json.loads(corpus.read_text())
            macro(pre["corpus"] + "Combos", fmt(c["combinations"]))
            macro(pre["corpus"] + "Valid", fmt(c["valid_combinations"]))
            macro(pre["corpus"] + "Dropped", fmt(c["combinations"] - c["valid_combinations"]))
            macro(pre["corpus"] + "Rows", fmt(c["train_rows"]))
            macro(pre["corpus"] + "Pool", fmt(c["pool_rows"]))
            log = pd.read_csv(ROOT / corpus_dir / "teacher_log.csv")
            first = log[log["attempt"] == 0]
            macro(pre["teacher"] + "FirstValid", fmt(100 * first["schema_valid"].mean(), 0))
            if "recovered" in log.columns:
                macro(pre["teacher"] + "FirstRecovered", fmt(100 * first["recovered"].mean(), 0))
        tm = ROOT / adapter_dir / "train_metrics.json"
        if tm.exists():
            t = json.loads(tm.read_text())
            macro(pre["train"] + "Steps", fmt(t["optimizer_steps"]))
            macro(pre["train"] + "Minutes", fmt(t["train_seconds"] / 60, 0))
            macro(pre["train"] + "PeakMem", fmt(t["peak_cuda_reserved_mb"], 0))
            macro(pre["train"] + "AdapterMb", fmt(t["adapter_mb"], 0))
            macro(pre["train"] + "Params", fmt(t["trainable_params"] / 1e6, 1))
            losses = [e["loss"] for e in t["loss_log"]]
            macro(pre["train"] + "LossFirst", fmt(losses[0], 3))
            macro(pre["train"] + "LossLast", fmt(losses[-1], 3))
            macro(pre["train"] + "Loss", fmt(t["train_loss"], 3))
            macro(pre["train"] + "Batch", f"{t.get('batch_size', 4)} / {t.get('grad_accum', 2)}")
        d = latest(pattern)
        if d is None:
            continue
        df = prepare(pd.read_csv(d / "requests.csv"))
        info = pd.read_csv(d / "models.csv").set_index("model")
        base = df[df["model"] == base_tag].sort_values("sample")
        train_file = ROOT / corpus_dir / "train.jsonl"
        seen = {json.loads(x)["instruction"] for x in train_file.read_text().splitlines() if x} \
            if train_file.exists() else set()
        df["seen"] = df.apply(lambda r: (
            f"State: [STATE_{r['state'].upper()}]\nObserved Events: "
            f"{r['tokens'] if isinstance(r['tokens'], str) and r['tokens'] else '[NO_EVENT]'}") in seen, axis=1)
        if rows:
            rows.append("\\midrule")
            split_rows.append("\\midrule")
        for model, which in ((base_tag, "base"), (ft_tag, "ft")):
            if model not in set(df["model"]):
                continue
            label = MODEL_KEYS[model][1]
            g = df[df["model"] == model].sort_values("sample")
            s = summarise(g)
            same = (g["action"].values == base["action"].values).mean()
            p = p_value(base["wall_ms"], g["wall_ms"]) if which == "ft" else float("nan")
            mem = info.loc[model, "ps_size_mb"] if model in info.index else float("nan")
            rows.append(
                f"{label} & {fmt(s['np'], 1)} & {fmt(s['prefill_med'], 0)} & {fmt(s['no'], 1)} & "
                f"{fmt(s['decode_med'], 0)} & {fmt(s['total_med'], 0)} & {fmt(s['total_p95'], 0)} & "
                f"{fmt(mem, 0)} & {pct(s['asvr'])} & {pct(s['capped'])} & "
                f"{pct(s['agree'])} & {fmt_p(p)} \\\\")
            fig_rows.append((label, s["prefill_med"], s["decode_med"], s["other_med"], s["total_med"]))
            mp = pre[which]
            for k, v, nd in (("Np", s["np"], 1), ("Prefill", s["prefill_med"], 0), ("No", s["no"], 1),
                             ("Decode", s["decode_med"], 0), ("Total", s["total_med"], 0),
                             ("Ptf", s["total_p95"], 0), ("Mem", mem, 0),
                             ("Asvr", 100 * s["asvr"], 0), ("Capped", 100 * s["capped"], 0),
                             ("Agree", 100 * s["agree"], 0), ("SameAsBase", 100 * same, 0),
                             ("Tps", s["tps"], 1), ("Recoverable", 100 * s["recoverable"], 0),
                             ("Energy", s["energy"], 1), ("Invented", 100 * s["invented"], 0)):
                macro(mp + k, fmt(v, nd))
            macro(mp + "P", fmt_p(p))
            FT_STATS[(key, which)] = {**s, "mem": mem, "requests": g}
            gi, bi = g.set_index("sample"), base.set_index("sample")
            cells = []
            for flag in (True, False):
                h = gi[gi["seen"] == flag]
                same_h = (h["action"] == bi.loc[h.index, "action"]).mean() if len(h) else float("nan")
                cells += [fmt(len(h)), pct(h["schema_valid"].mean()) if len(h) else "--",
                          pct(h["action_canonical"].astype(bool).mean()) if len(h) else "--",
                          pct(same_h) if which == "ft" else "--"]
                tag = "Seen" if flag else "Unseen"
                macro(mp + "Asvr" + tag, fmt(100 * h["schema_valid"].mean(), 0) if len(h) else "--")
                macro(mp + "N" + tag, fmt(len(h)))
            split_rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        b_med = df[df["model"] == base_tag]["wall_ms"].median()
        f_med = df[df["model"] == ft_tag]["wall_ms"].median() if ft_tag in set(df["model"]) else float("nan")
        macro(f"resFt{key}Red", fmt(100 * (b_med - f_med) / b_med, 0))
        frames.append(df)
    if not rows:
        return
    write_table("finetune", "\n".join(rows) + "\n")
    write_table("finetune_seen", "\n".join(split_rows) + "\n")
    stacked_latency_figure(fig_rows, "finetune_latency.pdf")
    pd.concat(frames).to_csv(FIGS / "finetune_requests_clean.csv", index=False)


# ── quantization chain (20 rows per variant) ──────────────────────────────────
CHAIN_ROWS = [  # family, label, adapter dir, variants: (FP16 merge, NF4->FP16 GGUF, NF4->Q8_0, NF4->Q4_K_M, NF4->Q4_0)
    ("Llama", "LLaMA 3.2 (3B)", "languagemodels/fine_tuned_llama3_lora",
     ("edgeiot-llama3.2-ft-fp16merge", None, "edgeiot-llama3.2-ft-q8", "edgeiot-llama3.2-ft-merged", None)),
    ("Qwen", "Qwen 2.5 (1.5B)", "languagemodels/fine_tuned_qwen25_lora",
     ("edgeiot-qwen2.5-ft-fp16merge", "edgeiot-qwen2.5-ft-f16", "edgeiot-qwen2.5-ft-q8", "edgeiot-qwen2.5-ft-merged", None)),
    ("Phi", "Phi-3.5 (3.8B)", "languagemodels/fine_tuned_phi35_lora",
     (None, None, "edgeiot-phi3.5-ft-q8", "edgeiot-phi3.5-ft-merged", "edgeiot-phi3.5-ft-q40")),
]


def chain():
    """Validity of each step of the quantization chain on the same 20 rows."""
    dirs = [d for d in OUT.glob("experiments/finetune-chain*") if (d / "requests.csv").exists()]
    if not dirs:
        return
    dirs.sort(key=lambda d: d.name.rsplit("_", 2)[-2] + d.name.rsplit("_", 1)[-1])  # by timestamp
    df = pd.concat([pd.read_csv(d / "requests.csv") for d in dirs])
    if "error" in df.columns:          # requests that failed (e.g. model did not load) are not answers
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    df = df.drop_duplicates(subset=["model", "sample"], keep="last")   # a later run of a tag wins
    valid = df.groupby("model")["schema_valid"].mean()
    names = ("Adapter", "FpMerge", "FsixteenGguf", "Qeight", "QfourKm", "QfourZero")
    lines = []
    for fam, label, adapter, variants in CHAIN_ROWS:
        diag = ROOT / adapter / "chain_diagnosis.json"
        nf4 = json.loads(diag.read_text()).get("nf4", {}).get("schema_valid") if diag.exists() else None
        cells = [nf4] + [valid.get(t) if t else None for t in variants]
        for name, v in zip(names, cells):
            macro(f"resChain{fam}{name}", fmt(100 * v, 0) if v is not None else "--")
        lines.append(f"{label} & " + " & ".join(pct(v) if v is not None else "--" for v in cells) + " \\\\")
    write_table("chain", "\n".join(lines) + "\n")
    macro("resChainRows", fmt(df.groupby("model").size().iloc[0]))


# ── worked example ─────────────────────────────────────
EXAMPLE_SAMPLE = 2   # an anomaly reading with three events on which the models disagree
ORDER = ["llama3.2:3b", "edgeiot-llama3.2-ft-merged", "qwen2.5:1.5b", "edgeiot-qwen2.5-ft-merged",
         "phi3.5:latest", "edgeiot-phi3.5-ft-merged"]


def e4_requests():
    frames, samples = [], None
    for _, _, _, pattern, _, _ in FT_PAIRS:
        d = latest(pattern)
        if d is None:
            continue
        frames.append(pd.read_csv(d / "requests.csv"))
        if samples is None and (d / "samples.csv").exists():
            samples = pd.read_csv(d / "samples.csv")
    return (pd.concat(frames) if frames else None), samples


# ── E4: which fine-tuned model is best? ──────────────────────────────────────
def mcnemar_p(a, b):
    """Exact two-sided McNemar test on paired booleans."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue


def ft_comparison():
    """Base and fine-tuned models compared on the same criteria; the best value per row is bold."""
    cols = [(k, w) for k, *_ in FT_PAIRS for w in ("base", "ft") if (k, w) in FT_STATS]
    if len(cols) < 2:
        return
    num = lambda name: float(macros[name].replace(",", "")) / 100 if macros.get(name, "--") != "--" else np.nan
    probe_key = {("Llama", "base"): "resProbeLlama", ("Qwen", "base"): "resProbeQwen", ("Phi", "base"): "resProbePhi",
                 ("Llama", "ft"): "resProbeFtMerged", ("Qwen", "ft"): "resProbeQwenFtMerged",
                 ("Phi", "ft"): "resProbePhiFtMerged"}
    stat = lambda f: {c: FT_STATS[c][f] for c in cols}
    probe_val = lambda group: {c: num(probe_key[c] + group) for c in cols}
    # (label, unit, values per column, better, decimals, percent?); a 1-tuple starts a group
    crit = [
        ("Speed and resources (E4, 100 rows)",),
        ("Median latency", "ms", stat("total_med"), "low", 0, False),
        ("95th-percentile latency", "ms", stat("total_p95"), "low", 0, False),
        ("Model memory", "MB", stat("mem"), "low", 0, False),
        ("Energy per decision", "J", stat("energy"), "low", 1, False),
        ("Answer quality (E4, 100 rows)",),
        ("Strict JSON (ASVR)", "", stat("asvr"), "high", 0, True),
        ("Recoverable action", "", stat("recoverable"), "high", 0, True),
        ("Invented actions", "", stat("invented"), "low", 0, True),
        ("Agreement with the rules", "", stat("agree"), "high", 0, True),
        ("Robustness (probe set, 22 cases)",),
        ("Policy cases", "", probe_val("Policy"), "high", 0, True),
        ("Unseen event types", "", probe_val("Unseen"), "high", 0, True),
        ("Free-text requests", "", probe_val("Freetext"), "high", 0, True),
        ("All probe cases", "", probe_val("All"), "high", 0, True),
    ]
    n_crit = sum(1 for c in crit if len(c) > 1)
    macro("resCmpCriteria", str(n_crit))
    rows = []
    wins = {scope: {c: 0 for c in cols} for scope in ("all", "base", "ft")}
    for c in crit:
        if len(c) == 1:
            if rows:
                rows.append("\\midrule")
            rows.append(f"\\multicolumn{{{len(cols) + 2}}}{{@{{}}l}}{{\\textit{{{c[0]}}}}} \\\\")
            continue
        label, unit, vals, better, nd, is_pct = c
        pick = min if better == "low" else max
        for scope in wins:
            sub = [v for col, v in vals.items() if scope in ("all", col[1]) and not np.isnan(v)]
            best = pick(sub)
            for col, v in vals.items():
                if scope in ("all", col[1]) and not np.isnan(v) and np.isclose(v, best):
                    wins[scope][col] += 1
        best = pick(v for v in vals.values() if not np.isnan(v))
        cells = []
        for col in cols:
            v = vals[col]
            txt = pct(v, nd) if is_pct else fmt(v, nd)
            cells.append(f"\\textbf{{{txt}}}" if not np.isnan(v) and np.isclose(v, best) else txt)
        rows.append(f"\\quad {label} & {unit} & " + " & ".join(cells) + " \\\\")
    rows += ["\\midrule",
             f"Best of all six (of {n_crit}) & & " + " & ".join(str(wins["all"][c]) for c in cols) + " \\\\",
             f"Best within its group (of {n_crit}) & & "
             + " & ".join(str(wins[c[1]][c]) for c in cols) + " \\\\"]
    write_table("ft_compare", "\n".join(rows) + "\n")
    for (k, w) in cols:
        macro(f"resWinsAll{k}{'Ft' if w == 'ft' else 'Base'}", str(wins["all"][(k, w)]))
        macro(f"resWins{'Ft' if w == 'ft' else 'Base'}{k}", str(wins[w][(k, w)]))

    # Paired tests between the fine-tuned models on the same evaluation rows
    req = {k: FT_STATS[(k, "ft")]["requests"].set_index("sample") for k, *_ in FT_PAIRS if (k, "ft") in FT_STATS}
    common = sorted(set.intersection(*(set(r.index) for r in req.values())))
    macro("resFtCmpN", fmt(len(common)))
    for a, b in (("Llama", "Qwen"), ("Llama", "Phi"), ("Qwen", "Phi")):
        if a in req and b in req:
            ra, rb = req[a].loc[common], req[b].loc[common]
            macro(f"resFtCmp{a}{b}AsvrP", fmt_p(mcnemar_p(ra["schema_valid"], rb["schema_valid"])))
            macro(f"resFtCmp{a}{b}LatP", fmt_p(p_value(ra["wall_ms"], rb["wall_ms"])))
            macro(f"resFtCmp{a}{b}SameAction", fmt(100 * (ra["action"].values == rb["action"].values).mean(), 0))

def worked_example():
    import sys
    sys.path.insert(0, str(ROOT))
    from edgeiot.semantic.actions import _first_json_object
    df, samples = e4_requests()
    if df is None or samples is None:
        return
    row = samples.iloc[EXAMPLE_SAMPLE]
    sensors = [("temperature", "Temperature", "\\si{\\celsius}"), ("humidity", "Humidity", "\\si{\\percent}"),
               ("noise_db", "Noise", "dB"), ("light_intensity", "Light", "lux"), ("motion_level", "Motion", "--")]
    lines = []
    for col, name, unit in sensors:
        lines.append(f"{name} & {unit} & {row['raw_' + col]:.2f} & {row['z_' + col]:+.2f} \\\\")
    write_table("example_input", "\n".join(lines) + "\n")
    macro("resExLocation", latex_escape(str(row["location"])))
    macro("resExBehaviour", latex_escape(str(row["behaviour"])))
    macro("resExState", latex_escape(str(row["state"])))
    macro("resExStateTok", latex_escape(f"[STATE_{str(row['state']).upper()}]"))
    macro("resExTokens", latex_escape(str(row["tokens"])))
    macro("resExAnomaly", "flagged" if row["anomaly"] == "ABNORMAL" else "not flagged")
    macro("resExRow", fmt(row["row_index"]))
    g = df[df["sample"] == EXAMPLE_SAMPLE].drop_duplicates("model").set_index("model")
    macro("resExRule", latex_escape(str(g.iloc[0]["rule_action"])))
    out = []
    for model in [m for m in ORDER if m in g.index]:
        r = g.loc[model]
        text = r["response"] if isinstance(r["response"], str) else ""
        obj = _first_json_object(text) or {}          # any JSON object, also with an invented action
        reason = obj.get("reasoning") if isinstance(obj.get("reasoning"), str) else text.strip().replace("\n", " ")
        cut = len(reason) > 150
        reason = latex_escape(reason[:147].rsplit(" ", 1)[0] if cut else reason) + ("\\ldots" if cut else "")
        action = r["action"] if isinstance(r["action"], str) else "(none)"
        valid = "\\checkmark" if r["schema_valid"] else "$\\times$"
        label = MODEL_KEYS[model][1].replace(" (merged)", "")
        out.append(f"{label} & {int(r['prompt_eval_count'])} & ``{reason}'' & "
                   f"\\texttt{{{latex_escape(action)}}} & {valid} & {fmt(r['wall_ms'], 0)} \\\\")
    write_table("example_answers", "\n\\midrule\n".join(out) + "\n")


# ── transcripts ──────────────────────────────────────────────────────────────
def transcripts(multi_df):
    if multi_df is None:
        return
    g = multi_df[(multi_df["model"] == "llama3.2:3b") & multi_df["schema_valid"]]
    picks = []
    for cond in (
        g["tokens"].str.contains("ANOMALY", na=False),
        (g["state"] == "sleeping"),
        g["tokens"].str.contains("TEMP_HIGH", na=False) & (g["state"] == "active"),
        g["tokens"].str.contains("LOW_LIGHT", na=False) & (g["state"] == "sedentary"),
        g["tokens"].fillna("").eq(""),
    ):
        cand = g[cond & ~g["sample"].isin([p["sample"] for p in picks])]
        if not cand.empty:
            picks.append(cand.iloc[0])
    lines = []
    for i, r in enumerate(picks, 1):
        obj = json.loads(r["response"])
        reason = obj["reasoning"]
        if len(reason) > 220:
            reason = reason[:217].rsplit(" ", 1)[0] + "\\ldots"
        tokens = r["tokens"] if isinstance(r["tokens"], str) and r["tokens"] else "[NO_EVENT]"
        lines.append(
            f"{i} & \\texttt{{[STATE\\_{r['state'].upper()}]}} \\newline \\texttt{{{latex_escape(tokens)}}} & "
            f"``{latex_escape(reason)}'' & \\texttt{{{latex_escape(str(r['action']))}}} & "
            f"\\texttt{{{latex_escape(r['rule_action'])}}} \\\\")
    write_table("transcripts", "\n\\midrule\n".join(lines) + "\n")


def probe_transcripts():
    """Rationales of selected probe answers (Table of Chapter 5)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from edgeiot.semantic.actions import extract_answer
    dirs = [d for d in sorted(OUT.glob("experiments/probe_*")) if (d / "requests.csv").exists()]
    if not dirs:
        return
    df = pd.concat([pd.read_csv(d / "requests.csv") for d in dirs]).drop_duplicates(
        subset=["model", "case"], keep="last")
    picks = [("llama3.2:3b", "U1"), ("llama3.2:3b", "N6"), ("phi3.5:latest", "U3"),
             ("llama3.2:3b", "P5"), ("qwen2.5:1.5b", "P8")]
    lines = []
    for model, case in picks:
        hit = df[(df["model"] == model) & (df["case"] == case)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        answer = extract_answer(r["response"]) or {}
        reason = answer.get("reasoning", "(no recoverable answer)")
        tokens = r["tokens"] if isinstance(r["tokens"], str) and r["tokens"] else "[NO_EVENT]"
        what = f"``{r['note']}''" if isinstance(r["note"], str) and r["note"] else tokens
        ok = "\\checkmark" if r["action"] in r["acceptable"].split("|") else "$\\times$"
        lines.append(
            f"{case} & {MODEL_KEYS[model][1].split(' (')[0]} & "
            f"\\texttt{{[STATE\\_{r['state'].upper()}]}} {latex_escape(what)} & "
            f"``{latex_escape(reason)}'' & \\texttt{{{latex_escape(str(r['action']))}}} {ok} \\\\")
    write_table("probe_transcripts", "\n\\midrule\n".join(lines) + "\n")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    pipeline()
    m = multi()
    tokenization()
    probe("experiments/probe*")
    finetune()
    chain()
    ft_comparison()
    worked_example()
    transcripts(m[0] if m else None)
    probe_transcripts()
    lines = ["% Generated by scripts/analyze_results.py — do not edit by hand"]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (TEX / "results_macros.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(macros)} macros written")


if __name__ == "__main__":
    main()
