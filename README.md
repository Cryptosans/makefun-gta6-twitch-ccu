# makefun-gta6-twitch-ccu

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Cryptosans/makefun-gta6-twitch-ccu/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Cryptosans/makefun-gta6-twitch-ccu/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), from the first hour's viewer count — as five quantiles an hour, scored by pinball loss on the log scale. Built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty, $250 in USDC.

## Method

Launch-curve regression in log space, fit fresh on every case from the comparables the case ships:

- Each comparable launch gives a release-day curve of log ratios `ln(viewers_h / viewers_1)` for hours 2–24. Early-access-contaminated launches are dropped — their release day is not a clean launch day.
- The shape of a launch day is driven by how much of the world is already unlocked and awake in hour 1: across the comparables, the evening ramp correlates about −0.94 with `ln(hour-1 print)`. So for each hour, a weighted least-squares line is fit, `ratio_h ~ a_h + b_h · ln(p1)`, weighted by a Gaussian kernel on `ln(p1)` around the case's own hour-1 print (bandwidth 1.0). The prediction clamps `ln(p1)` to the comparables' range, so a record-size hour 1 never extrapolates past the data.
- The spread is the kernel-weighted residual standard deviation of that hour's fit, floored at 0.12 and capped at 0.60 log units; quantiles are `median · exp(z_tau · sigma_h)`. With too few usable comparables it falls back to persistence at the hour-1 print with a widening spread.

Practice score (local, against the historical SullyGnome prints): mean pinball loss **0.160** — p00 0.165, p01 0.158, p02 0.125, p03 0.191 — vs the house baseline's 0.387. Deterministic, ~0.2 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` in the input directory and writes `<id>.out` — 23 rows, hours 2–24, five quantiles each — to the output directory. No network, no randomness.

---

<sub>Entered on [make.fun](https://make.fun/?ref=Cryptosans&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
