from __future__ import annotations
from pathlib import Path

import statistics, math
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


def _read_sysfs(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8", "replace").strip("\x00").strip()
    except OSError:
        return None

# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    times = []
    for _ in range(repeats):
        start= bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        times.append((end-start)/1_000_000.0)
    return times




def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("find_warmup_boundary", "fewer than 4 samples")
    
    second_half = samples[len(samples) // 2:]
    median = statistics.median(second_half)
    if median <= 0:
        return unknown("find_warmup_boundary", "settled median is not positive")
    
    threshold = median * (1 + WARMUP_TOL)
    
    discarded = 0
    for s in samples:
        if s > threshold:
            discarded += 1
        else:
            break
    
    return measured(
        discarded,
        "leading prefix above (1 + 0.5) x median of the run's second half",
        settled_rate_ms=round(median,4),
        threshold_ms=round(threshold,4),
        tolerance=WARMUP_TOL,
        retained=len(samples)-discarded
    )


def _percentile(sorted_s: list[float], q: float) -> float:
    n = len(sorted_s)
    h = (n-1) * q /100
    i = math.floor(h)
    if i + 1 < n:
        return sorted_s[i] + (h - i) * (sorted_s[i + 1] - sorted_s[i])
    return sorted_s[i]

def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "p50": None, "p95": None, "p99": None}
    
    s = sorted(samples)
    n = len(s)
    mean = statistics.fmean(s)
    std = statistics.stdev(s) if n > 2 else 0.0

    result = {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4)
    }
    for q in PERCENTILES:
        result[f"p{q}"] = round(_percentile(s,q),4)
    return result

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    n = len(samples)
    if n < MIN_SAMPLES_FOR_MODALITY:
        return unknown("is_multimodal", f"fewer than {MIN_SAMPLES_FOR_MODALITY} samples")
    s = sorted(samples)
    trim = int(n*0.05)
    trimmed = s[trim:n-trim]

    gaps = [trimmed[i+1] - trimmed[i] for i in range(len(trimmed)-1)]
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return unknown("is_multimodal", "timer resolution too coarse (median gap is zero)")

    widest_gap = max(gaps)
    widest_idx = gaps.index(widest_gap)
    ratio = widest_gap / median_gap

    split_value = trimmed[widest_idx]
    left = [x for x in samples if x <= split_value]
    right = [x for x in samples if x > split_value]

    both_sides_big = (len(left) >= MIN_MODE_FRACTION * n and len(right) >= MIN_MODE_FRACTION * n)
    value = ratio >= MULTIMODAL_GAP_RATIO and both_sides_big

    return measured(
        value,
        "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side",
        gap_ratio=round(ratio,2),
        widest_gap_ms=round(widest_gap,3),
        typical_gap_ms=round(median_gap,5),
        modes=[
            {"n": len(left), "share": round(len(left)/n, 2),
            "median_ms": round(statistics.median(left),4)},
            {"n": len(right), "share": round(len(right)/n, 2),
            "median_ms": round(statistics.median(right),4)}
        ]
    )




# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel","-q"])
    if not result.ok or result.returncode != 0:
        return unknown(
            result.source,
            f"nvpmodel failed (needs sudo?): {result.error or 'exit ' + str(result.returncode)}"
        )
    lines = result.stdout.splitlines()
    mode_name, mode_index = None, None
    for idx, line in enumerate(lines):
        if "NV Power Mode:" in line:
            mode_name = line.split(":",1)[1].strip()
            if idx + 1 < len(lines):
                try:
                    mode_index = int(lines[idx + 1].strip())
                except ValueError:
                    mode_index = None
            break
    
    fmin = read_text(bench.telemetry, CPUFREQ_MIN)
    fmax = read_text(bench.telemetry, CPUFREQ_MAX)
    if fmin is None or fmax is None:
        jetson_clocks = None
        clocks_src = unknown(f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}", "could not read one or both cpufreq files") 
    else:
        jetson_clocks = fmin == fmax
        clocks_src = measured(f"scaling_min_freq={fmin}, scaling_max_freq={fmax}", f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}")

    return measured(mode_name, result.source, mode_index=mode_index, jetson_clocks=jetson_clocks, jetson_clocks_source=clocks_src)


def probe_telemetry(bench: Bench) -> dict[str, Any]:
   base = Path(bench.telemetry) / THERMAL_ZONES
    zones_read, hottest, hottest_dir = 0, None, None
    for zone_dir in sorted(base.glob("thermal_zone*")):
        raw_txt = _read_sysfs(zone_dir / "temp")
        if not raw_txt:
            continue
        try:
            raw = int(raw_txt)
        except ValueError:
            continue
        zones_read += 1
        if raw <= -1000:
            continue
        temp_c = raw / 1000.0
        if hottest is None or temp_c > hottest:
            hottest, hottest_dir = temp_c, zone_dir.name

    if hottest is None:
        temperature = unknown(f"{THERMAL_ZONES}/*/temp", "no readable thermal zones")
    else:
        zname = _read_sysfs(base / hottest_dir / "type")
        temperature = measured(round(hottest, 2), f"{THERMAL_ZONES}/*/temp",
                               zone=zname, zones_read=zones_read)
    
    found = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if found is None:
        power = unknown(" | ".join(POWER_RAIL_CANDIDATES), "none of the documented INA3221 rail paths could be read")
    else:
        ppath, ptext = found
        power = measured(int(ptext), ppath)
    
    found = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if found is None:
        gpu = unknown(" | ".join(GPU_LOAD_CANDIDATES), "no GPU load file found")
    else:
        gpath, gtext = found
        gpu = measured(round(int(gtext) / 10.0, 1), gpath, units = "per-mille / 10")

    return {"temperature_c": temperature, "power_mw": power, "gpu_utilization_percent": gpu}

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)