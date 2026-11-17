"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (make.fun
gta6-twitch-ccu-forecast-01).

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output): for hours 2..24, five
quantiles of that hour's concurrent-viewer count.

Method: launch-curve regression in log space.
  - Every comparable launch in the case ships its release-day hourly curve
    (14 days before to 3 days after its release). For each comparable we take
    the log ratio r_h = ln(viewers_h / viewers_1) for hours 2..24.
  - The shape of a launch day is driven by how much of the world is already
    unlocked and awake in hour 1: across the comparables, the evening ramp
    correlates about -0.94 with ln(hour-1 print). So for each hour h we fit a
    weighted least-squares line r_h ~ a_h + b_h * ln(p1), weighted by a
    Gaussian kernel on ln(p1) around the case's own hour-1 print
    (bandwidth 1.0 in log units). The case's prediction uses its own ln(p1),
    clamped to the comparables' range so a record-size hour 1 never
    extrapolates the line past the data.
  - Early-access-contaminated comparables (their release day is not a clean
    launch day) are dropped before fitting.
  - The spread is the kernel-weighted residual standard deviation of that
    hour's fit, floored at 0.12 and capped at 0.60 log units: the five
    quantiles are median * exp(z_tau * sigma_h).
  - Fallback: with too few usable comparables, persistence at the hour-1
    print with a widening random-walk spread (the house baseline's shape).

Python 3.13, standard library only. Deterministic: no randomness, no
network, no clock reads. One case runs in well under a second.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Gaussian kernel bandwidth on ln(hour-1 print).
BANDWIDTH = 1.0
# Spread of an hour's log-ratio forecast, in log units.
SIGMA_FLOOR = 0.12
SIGMA_CAP = 0.60
# A comparable needs a real hour-1 print and at least this many scored hours.
MIN_HOURS = 20
# Fallback spread (house baseline's random-walk sigma).
FALLBACK_SIGMA = 0.15


def parse_time(text):
    """ISO 8601 UTC timestamp -> datetime (aware)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up (the house baseline's convention)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def solve2(a11, a12, a22, c1, c2):
    """Solve a 2x2 symmetric linear system; None when singular."""
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return None
    return ((c1 * a22 - c2 * a12) / det, (a11 * c2 - a12 * c1) / det)


def weighted_line(points):
    """Weighted least squares y ~ a + b*x.

    points: iterable of (x, y, w). Returns (a, b, weighted residual sd)
    or None when the fit is degenerate (too little total weight, or a
    singular system).
    """
    sw = sx = sy = sxx = sxy = 0.0
    for x, y, w in points:
        sw += w
        sx += w * x
        sy += w * y
        sxx += w * x * x
        sxy += w * x * y
    if sw < 1e-9:
        return None
    fit = solve2(sw, sx, sxx, sy, sxy)
    if fit is None:
        return None
    a, b = fit
    var = 0.0
    for x, y, w in points:
        r = y - (a + b * x)
        var += w * r * r
    return a, b, math.sqrt(var / sw)


def release_day_ratios(comp):
    """A comparable's release-day log ratios, keyed by hour 2..24.

    Returns (hour1_print, {hour: ln(viewers_h / viewers_1)}) or None when
    the comparable has no usable release day.
    """
    day_start = parse_time(comp["day_start"])
    by_hour = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            at = parse_time(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (at - day_start).total_seconds() / 3600.0
        if 0.0 <= offset < 24.0:
            by_hour[int(offset) + 1] = float(viewers)
    if 1 not in by_hour or len(by_hour) < MIN_HOURS:
        return None
    p1 = by_hour[1]
    return p1, {h: math.log(v / p1) for h, v in by_hour.items() if h >= 2}


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels: %r" % (case.get("levels"),))
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    # Collect usable comparables, dropping early-access-contaminated ones:
    # their release day is not a clean launch day.
    comps = []
    for comp in case.get("comparables") or []:
        unlock = comp.get("unlock") or ""
        if "early-access-contaminated" in unlock:
            continue
        parsed = release_day_ratios(comp)
        if parsed is not None:
            comps.append(parsed)

    hours = []
    if len(comps) < 8:
        # Too little to fit: persistence with a widening spread.
        for row in case["forecast"]:
            h = row["hour"]
            qs = [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)))
                  for z in Z]
            hours.append({"hour": h, "q": qs})
        return {"hours": hours}

    log_p1s = [math.log(p) for p, _ in comps]
    lo, hi = min(log_p1s), max(log_p1s)
    # Clamp: never extrapolate the fitted line past the comparables.
    log_p1_eff = max(lo, min(hi, log_p1))

    for row in case["forecast"]:
        h = row["hour"]
        points = []
        for p_comp, ratios in comps:
            if h in ratios:
                w = math.exp(-0.5 * ((math.log(p_comp) - log_p1) / BANDWIDTH) ** 2)
                points.append((math.log(p_comp), ratios[h], w))
        fit = weighted_line(points) if len(points) >= 8 else None
        if fit is None:
            qs = [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)))
                  for z in Z]
        else:
            a, b, sigma = fit
            # The fitted log-ratio at the case's (clamped) hour-1 print.
            # Bound it: no hour of a launch day plausibly sits below 5% or
            # above ~33x its first hour, so a wild slope can never drag a
            # quantile outside the format or into nonsense.
            mu = max(-3.0, min(3.5, a + b * log_p1_eff))
            sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
            qs = [round2(p1 * math.exp(mu + z * sigma)) for z in Z]
        # Keep every quantile inside the format's bounds and ordered.
        qs = [min(1_000_000_000.0, max(1.0, float(q))) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
