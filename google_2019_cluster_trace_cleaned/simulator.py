"""Discrete-event simulator of the reject-and-retry model (thesis Ch. 3).

Event logic, measurement windows and RNG consumption order are identical to
``simulate_external_steady_state`` in the original script, so with the same
seed and deterministic/exponential distributions the outputs match exactly.
The processing-time and restart-delay distributions are now pluggable.
"""
from __future__ import annotations

import heapq
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from .distributions import Distribution

_ARRIVAL, _START, _COMMIT = 0, 1, 2


def simulate(
    lambda_ext: float,
    proc: Distribution,
    restart: Distribution,
    alpha: float,
    warmup_time: float,
    measure_time: float,
    seed: int,
    max_events: int,
) -> Dict[str, Any]:
    """One steady-state run.

    External Poisson(lambda_ext) arrivals start an attempt immediately.  At each
    successful commit, every attempt in flight that started before it is
    disrupted independently w.p. alpha; an attempt that saw m intervening
    successful commits is accepted w.p. (1 - alpha)^m.  A rejected task waits a
    restart delay and starts a new attempt with a fresh processing time.

    Rates (G, S) are measured over [warmup, warmup + measure).  Tasks arriving in
    that window are tagged and followed to their first successful commit, with
    external arrivals continuing until all are done or ``max_events`` is hit
    (status ``max_events_exceeded``; all metrics NaN).
    """
    wall0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    draw_proc = proc.sample
    draw_restart = restart.sample
    measure_start = float(warmup_time)
    measure_end = float(warmup_time + measure_time)

    pq: List[Tuple[float, int, int, Any]] = []
    push, pop = heapq.heappush, heapq.heappop
    seq = 0
    task_id = 0
    n_events = 0

    original_arrival: Dict[int, float] = {}
    measured_active = set()
    measured_total = 0
    global_succ_count = 0
    offered_in_window = 0
    succ_in_window = 0
    delays: List[float] = []

    push(pq, (float(rng.exponential(1.0 / lambda_ext)), seq, _ARRIVAL, None))
    seq += 1
    time_evt = 0.0

    while pq:
        n_events += 1
        if n_events > max_events:
            return {
                "lambda": lambda_ext,
                "S_hat": np.nan, "G_hat": np.nan, "p_v_hat": np.nan, "W_hat": np.nan,
                "n_delay_samples": len(delays),
                "measured_tasks": measured_total,
                "status": "max_events_exceeded",
                "n_events": n_events - 1,
                "sim_end_time": time_evt,
                "n_unfinished_tagged": len(measured_active),
                "wall_seconds": time.perf_counter() - wall0,
                "raw_delays": np.asarray(delays, dtype=float),
            }

        time_evt, _, etype, payload = pop(pq)

        if time_evt > measure_end and not measured_active:
            break

        if etype == _ARRIVAL:
            a = float(time_evt)
            tid = task_id
            task_id += 1
            original_arrival[tid] = a
            push(pq, (a, seq, _START, tid))
            seq += 1
            if measure_start <= a < measure_end:
                measured_active.add(tid)
                measured_total += 1
            if a < measure_end or measured_active:
                push(pq, (a + float(rng.exponential(1.0 / lambda_ext)), seq, _ARRIVAL, None))
                seq += 1
            continue

        if etype == _START:
            tid = int(payload)
            commit_t = float(time_evt) + draw_proc(rng)
            push(pq, (commit_t, seq, _COMMIT, (tid, float(time_evt), global_succ_count)))
            seq += 1
            if measure_start <= time_evt < measure_end:
                offered_in_window += 1
            continue

        # Commit request.
        current_time = float(time_evt)
        tid, start_time, start_succ = payload
        k = global_succ_count - int(start_succ)
        survival_prob = (1.0 - alpha) ** k
        if rng.random() >= survival_prob:
            push(pq, (current_time + draw_restart(rng), seq, _START, tid))
            seq += 1
        else:
            global_succ_count += 1
            if measure_start <= current_time < measure_end:
                succ_in_window += 1
            if tid in measured_active:
                delays.append(current_time - original_arrival[tid])
                measured_active.remove(tid)
            original_arrival.pop(tid, None)

    obs_time = measure_end - measure_start
    return {
        "lambda": lambda_ext,
        "S_hat": succ_in_window / obs_time if obs_time > 0 else np.nan,
        "G_hat": offered_in_window / obs_time if obs_time > 0 else np.nan,
        "p_v_hat": succ_in_window / offered_in_window if offered_in_window else np.nan,
        "W_hat": float(np.mean(delays)) if delays else np.nan,
        "n_delay_samples": len(delays),
        "measured_tasks": measured_total,
        "status": "ok",
        "n_events": n_events,
        "sim_end_time": time_evt,
        "n_unfinished_tagged": 0,
        "wall_seconds": time.perf_counter() - wall0,
        "raw_delays": np.asarray(delays, dtype=float),
    }
