
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    clean = []
    for x in ms:
        f = _finite(x, "latency sample")
        if f < 0:
            raise AgreementError("Sample is negative")
        clean.append(f)
    if len(clean) == 0:
        raise AgreementError("No samples")
    n = abs(len(clean))
    mean = 1/n * sum(clean)
    worst = max(clean)

    pct = {}
    for q in LATENCY_PERCENTILES:
        pct[f"p{q}"] = _percentile(clean, q)
    
    jitter = pct["p95"] - pct["p50"]

    return {"n": n, "mean": mean, "worst": worst, **pct, "jitter": jitter}



def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    if _finite(base_ms, "latency sample") <= 0:
        raise AgreementError("base_ms must be larger than 0")
    if _finite(other_ms, "latency sample") <= 0:
        raise AgreementError("other_ms must be larger than 0")
    
    factor = base_ms / other_ms
    faster = factor > 1.0
    saved_ms = base_ms - other_ms

    return {"factor": factor, "faster": faster, "saved_ms": saved_ms}


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    if not a or not b or len(a) != len(b):
        raise AgreementError("a or b are empty or not equal length")

    n = len(a)
    for x in a:
        _finite(x, "a")
    for x in b:
        _finite(x, "b")

    best_i = 0
    best = abs(a[0] - b[0])
    for i in range(1, n):
        d = abs(a[i] - b[i])
        if d > best:
            best = d
            best_i = i
    
    return {"max_abs": best, "index": best_i, "a": a[best_i], "b": b[best_i], "n": n}

def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    if not a or not b or len(a) != len(b):
        raise AgreementError("a or b are empty or not equal length")
    if floor <= 0:
        raise AgreementError("floor must be strictly positive")
    n = len(a)
    for x in a:
        _finite(x, "a")
    for x in b:
        _finite(x, "b")

    best_i = 0
    denom_min = abs(b[0])
    floored = 1 if abs(b[0]) < floor else 0
    best = abs(a[0] - b[0]) / max(abs(b[0]), floor)
    for i in range(1, n):
        d = abs(b[i])
        denom_min = min(denom_min, d)
        if d < floor:
            floored+= 1
        r = abs(a[i] - b[i]) / max(d, floor)
        if r > best:
            best = r
            best_i = i
    
    return {"max_rel": best, "index": best_i, "floored": floored, "denom_min": denom_min, "n": n}

def _argmax(row: Sequence[float]) -> int:
    best_i = 0
    best = row[0]
    for c in range(1, len(row)):
        if row[c] > best:
            best = row[c]
            best_i = c
    return best_i

def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    n = len(rows_a)
    if n == 0 or n != len(rows_b):
        raise AgreementError("row counts are zero or unequal")
    
    agree = 0
    first_disagreement = None
    for k in range(n):
        ra = rows_a[k]
        rb = rows_b[k]
        if not ra or not rb or len(ra) != len(rb):
            raise AgreementError("a row is empty or widths differ")
        for x in ra:
            _finite(x, "rows_a")
        for x in rb:
            _finite(x, "rows_b")
        ia = _argmax(ra)
        ib = _argmax(rb)
        if ia == ib:
            agree += 1
        if ia != ib and first_disagreement is None:
            first_disagreement = k
        
    disagree = n - agree
    rate = agree/n

    return {"n": n, "agree": agree, "disagree": disagree, "rate": rate, "first_disagreement": first_disagreement}


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise AgreementError("no rows")
    if tol < 0:
        raise AgreementError("tol must be non-negative")
    margins = []
    for r in rows:
        if len(r) < 2:
            raise AgreementError("a row has fewer than 2 classes")
        for x in r:
            _finite(x, "rows")
        s = sorted(r, reverse=True)
        margins.append(s[0] - s[1])

    best_i = 0
    best = margins[0]
    for i in range(1, n):
        if margins[i] < best:
            best = margins[i]
            best_i = i

    median = _percentile(margins, 50)
    mean = sum(margins)/n
    below_tol = sum(1 for m in margins if m < tol)
    safe = below_tol == 0

    return {"n": n, "min": best, "median": median, "mean": mean, "argmin": best_i, "below_tol": below_tol, "safe": safe}

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()