import statistics
from typing import Any

try:
    from bench import measured, unknown
    from measure import STATIONARITY_TOL, MIN_SAMPLES_FOR_STATIONARITY
except ImportError:
    def unknown(source, why):
        return {"value": None, "source": source, "status": "unknown", "detail": why}
    def measured(value, source, **extra):
        return {"value": value, "source": source, "status": "measured", **extra}
    MIN_SAMPLES_FOR_STATIONARITY = 12
    STATIONARITY_TOL = 0.1



def is_stationary(samples: list[float]) -> dict[str, Any]:
    n = len(samples)
    if n < MIN_SAMPLES_FOR_STATIONARITY:
        return unknown("is_stationary", "There are too few samples to divide into thirds")
    
    sample_median = statistics.median(samples)
    if sample_median <= 0:
        return unknown("is_stationary", "The median is negative")
    
    k = n // 3
    first_third = statistics.median(samples[:k])
    last_third = statistics.median(samples[n-k:])

    delta = last_third - first_third
    relative_drift = abs(delta) / sample_median

    direction = None

    if delta > 0:
        direction = "slower"
    elif delta < 0:
        direction = "faster"
    else: 
        direction = "flat"

    return measured(
        relative_drift <= STATIONARITY_TOL,
        "last-third vs first-thir median, drift <= 10% of the run median",
        first_third_median_ms=round(first_third,4),
        last_third_median_ms=round(last_third,4),
        drift_ms=round(delta,4),
        drift_relative=round(relative_drift,4),
        direction=direction,
        tolerance=STATIONARITY_TOL
    )