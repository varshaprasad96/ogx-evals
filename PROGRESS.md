# Progress Logging

This patch adds progress reporting and per-config server log files so long
A-D runs are easier to follow.

## What changed

- `scripts/run_experiment.py`, `scripts/run_injection_probes.py`,
  `scripts/ingest_data.py`: emit progress lines with elapsed time, ETA,
  queries-per-minute, and cumulative error count. Tune cadence with
  `--progress-every N` (default `10`).
- `run_all.sh`: prefixes each stage with a `[HH:MM:SS]` timestamp, redirects
  background auth and OGX server logs to `logs/config_<X>_auth.log` and
  `logs/config_<X>_ogx.log`, and reports per-config elapsed time.
- Fixes a `PIDS[@]: unbound variable` error in the cleanup trap when no
  background processes were started.

## Example output

```
[00:25:18]   Running experiment...
  [Authorized] Starting 900 queries (300 queries x 3 runs)
  [Authorized] 100/900 ( 11.1%) elapsed=5m23s eta=43m11s rate=18.5/min errors=0
  [Authorized] 200/900 ( 22.2%) elapsed=11m53s eta=41m35s rate=16.8/min errors=0
  ...
  [Probes] 50/900 (  5.6%) elapsed=7m18s eta=2h04m14s rate=6.8/min errors=0
```

## Usage

The flag is exposed on every per-script invocation:

```bash
uv run python scripts/run_experiment.py --config B --progress-every 5
uv run python scripts/run_injection_probes.py --config B --progress-every 5
uv run python scripts/ingest_data.py --config B
```

`run_all.sh` passes `--progress-every 10` to the experiment scripts by
default. To investigate server behavior, tail the per-config logs:

```bash
tail -f logs/config_B_ogx.log
tail -f logs/config_B_auth.log
```
