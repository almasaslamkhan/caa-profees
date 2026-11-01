import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

from cdrc.profees_coding.deploy.api.model.app_gen2 import ProfeesModel
from cdrc.profees_coding.deploy.api.utils import DeliveryVehicle, InputData

# Suppress noisy metrics logging when local config is unavailable.
logging.getLogger("aoa_agent.profees_coding_agent.agent_obs.metrics").setLevel(logging.CRITICAL)


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


model = ProfeesModel(
    delivery_vehicle=DeliveryVehicle.CAAM_V2,
    mode="SDS",
    log_level="info",
)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def load_input(payload_path: Path) -> InputData:
    payload = json.loads(payload_path.read_text())
    return InputData(**payload)


async def run_prediction(inp: InputData) -> float:
    t0 = time.perf_counter()
    # with suppress_stderr():
    await model.predict(inp)
    return time.perf_counter() - t0


def percentile(sorted_vals, p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


async def benchmark_payload(
    name: str,
    payload_path: Path,
    n: int = 100,
    warmup: int = 0,
):
    print(f"\n===== Benchmark: {name} =====")
    print(f"Payload: {payload_path}")

    inp = load_input(payload_path)

    # Warmup (excluded from stats)
    for _ in range(warmup):
        await run_prediction(inp)

    times = []
    for i in range(n):
        t = await run_prediction(inp)
        times.append(t)

    times_sorted = sorted(times)

    print(f"calls: {n} (warmup: {warmup})")
    print(f"mean: {mean(times):.6f} s")
    print(f"std:  {stdev(times):.6f} s")
    print(f"min:  {times_sorted[0]:.6f} s")
    print(f"p50:  {percentile(times_sorted, 50):.6f} s")
    print(f"p95:  {percentile(times_sorted, 95):.6f} s")
    print(f"p99:  {percentile(times_sorted, 99):.6f} s")
    print(f"max:  {times_sorted[-1]:.6f} s")

    return times


async def benchmark_payload_concurrent(
    name: str,
    payload_path: Path,
    concurrency: int = 3,
    n_batches: int = 100,
    warmup_batches: int = 0,
):
    print(f"\n===== Concurrent Benchmark: {name} | concurrency={concurrency} =====")
    print(f"Payload: {payload_path}")

    inp = load_input(payload_path)

    async def one_batch(batch_idx: int):
        # measure wall time for the batch + per-call times
        t0 = time.perf_counter()
        results = await asyncio.gather(*[run_prediction(inp) for _ in range(concurrency)])
        wall = time.perf_counter() - t0
        return wall, results

    # Warmup (excluded)
    for _ in range(warmup_batches):
        await one_batch(-1)

    wall_times = []
    per_call_times = []

    for b in range(n_batches):
        wall, results = await one_batch(b)

        wall_times.append(wall)
        # keep only successful per-call times
        per_call_times.extend([t for t in results if t is not None])

    # ---- Report wall-time stats (batch latency)
    wall_sorted = sorted(wall_times)
    print("\n-- Wall time per batch (seconds) --")
    print(f"batches: {n_batches} (warmup_batches: {warmup_batches})")
    print(f"mean: {mean(wall_times):.6f} s")
    print(f"std:  {stdev(wall_times):.6f} s" if len(wall_times) >= 2 else "std:  0.000000 s")
    print(f"min:  {wall_sorted[0]:.6f} s")
    print(f"p50:  {percentile(wall_sorted, 50):.6f} s")
    print(f"p95:  {percentile(wall_sorted, 95):.6f} s")
    print(f"p99:  {percentile(wall_sorted, 99):.6f} s")
    print(f"max:  {wall_sorted[-1]:.6f} s")

    # ---- Report per-call stats (individual latency under concurrency)
    if len(per_call_times) >= 2:
        call_sorted = sorted(per_call_times)
        total_attempts = n_batches * concurrency
        print("\n-- Per-call latency under concurrency (seconds) --")
        print(f"successful calls: {len(per_call_times)} / {total_attempts}")
        print(f"mean: {mean(per_call_times):.6f} s")
        print(f"std:  {stdev(per_call_times):.6f} s")
        print(f"min:  {call_sorted[0]:.6f} s")
        print(f"p50:  {percentile(call_sorted, 50):.6f} s")
        print(f"p95:  {percentile(call_sorted, 95):.6f} s")
        print(f"p99:  {percentile(call_sorted, 99):.6f} s")
        print(f"max:  {call_sorted[-1]:.6f} s")
    else:
        print("\nNot enough successful per-call samples to compute per-call statistics.")

    return wall_times, per_call_times


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
async def main():
    payload_code = Path("cdrc/profees_coding/deploy/api/model/payload.json")
    payload_recode = Path("cdrc/profees_coding/deploy/api/model/payload_recode.json")

    await benchmark_payload(
        name="CODE",
        payload_path=payload_code,
        n=25,
        warmup=0,
    )

    await benchmark_payload(
        name="RECODE",
        payload_path=payload_recode,
        n=25,
        warmup=0,
    )

    # After sequential benchmarks, measure 3 concurrent invokes
    await benchmark_payload_concurrent(
        name="CODE",
        payload_path=payload_code,
        concurrency=3,
        n_batches=25,
        warmup_batches=0,
    )

    await benchmark_payload_concurrent(
        name="RECODE",
        payload_path=payload_recode,
        concurrency=3,
        n_batches=25,
        warmup_batches=0,
    )


if __name__ == "__main__":
    asyncio.run(main())
