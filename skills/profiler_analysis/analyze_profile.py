#!/usr/bin/env python3
"""
Neuron Profiler JSON Feature Engineering Tool

Transforms raw Neuron Profiler JSON into pre-computed, structured analysis
that an LLM can directly reason about.

Usage:
    python analyze_profile.py <report_dir> [-o output.json]

The tool auto-detects neff_*_full.json files in the report directory,
skipping system_*.json files.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_engine(name):
    """Normalize engine names to lowercase (instruction.subgroup uses TitleCase)."""
    return name.lower() if name else name


def ts_to_ns(ts, ticks_per_ns):
    """Convert timestamp ticks to nanoseconds."""
    return ts / ticks_per_ns if ticks_per_ns else ts


def flatten_dma_location(loc):
    """Flatten nested DMA source/dest like [['INPUT']] to 'INPUT'."""
    if isinstance(loc, list):
        while isinstance(loc, list) and len(loc) == 1:
            loc = loc[0]
        if isinstance(loc, list):
            return "/".join(str(x) for x in loc)
    return str(loc)


def intervals_overlap(s1, e1, s2, e2):
    """Return overlap duration between two intervals, 0 if none."""
    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)
    return max(0, overlap_end - overlap_start)


# ---------------------------------------------------------------------------
# Dimension 1: Executive Summary
# ---------------------------------------------------------------------------


def extract_summary(data):
    """Extract executive summary from summary[0]."""
    if not data.get("summary"):
        return {}
    s = data["summary"][0]
    meta = data["metadata"][0] if data.get("metadata") else {}

    engine_keys = ["tensor", "vector", "scalar", "gpsimd"]
    engines = {}
    for eng in engine_keys:
        engines[eng] = {
            "active_time_s": s.get(f"{eng}_engine_active_time", 0),
            "active_time_pct": s.get(f"{eng}_engine_active_time_percent", 0),
            "instruction_count": s.get(f"{eng}_engine_instruction_count", 0),
        }
    engines["sync"] = {
        "instruction_count": s.get("sync_engine_instruction_count", 0),
    }
    engines["dma"] = {
        "active_time_s": s.get("dma_active_time", 0),
        "active_time_pct": s.get("dma_active_time_percent", 0),
    }

    total_time_s = s.get("total_time", 0)
    total_time_ns = total_time_s * 1e9

    hbm_read = s.get("hbm_read_bytes", 0)
    hbm_write = s.get("hbm_write_bytes", 0)
    hbm_bw = meta.get("hbm_ddr_bandwidth", 0)
    peak_flops = (
        meta.get("pe_clock_freq", 0)
        * 1e9
        * meta.get("pe_num_rows", 128)
        * meta.get("pe_num_cols", 128)
    )

    # Arithmetic intensity = flops / bytes
    # We approximate from tensor engine time and HBM bytes
    tensor_time_s = s.get("tensor_engine_active_time", 0)
    approx_flops = tensor_time_s * peak_flops if tensor_time_s else 0
    total_hbm_bytes = hbm_read + hbm_write
    arithmetic_intensity = approx_flops / total_hbm_bytes if total_hbm_bytes > 0 else 0
    peak_ratio = peak_flops / hbm_bw if hbm_bw > 0 else 0
    bound_type = (
        "compute-bound" if arithmetic_intensity > peak_ratio else "memory-bound"
    )

    return {
        "total_time_ns": total_time_ns,
        "total_active_time_ns": s.get("total_active_time", 0) * 1e9,
        "active_time_pct": (
            s.get("total_active_time", 0) / total_time_s * 100
            if total_time_s > 0
            else 0
        ),
        "engines": engines,
        "mfu_pct": s.get("mfu_estimated_percent", 0),
        "mbu_pct": s.get("mbu_estimated_percent", 0),
        "hfu_pct": s.get("hfu_estimated_percent", 0),
        "hbm_read_bytes": hbm_read,
        "hbm_write_bytes": hbm_write,
        "spill_bytes": s.get("sbuf_read_bytes", 0),
        "reload_bytes": s.get("sbuf_write_bytes", 0),
        "instance_type": s.get("instance_type", ""),
        "bound_type": bound_type,
        "arithmetic_intensity": round(arithmetic_intensity, 2),
        "peak_flops_bandwidth_ratio": round(peak_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Dimension 2+4: Multi-Scale Time Window Heatmap + Phase Detection
# ---------------------------------------------------------------------------

PHASE_THRESHOLDS = {
    "active": 10,  # % threshold to consider an engine "active" in a window
}


def classify_phase(tensor_pct, vector_pct, scalar_pct, gpsimd_pct, dma_pct):
    """Classify a time window into a phase label."""
    compute_active = any(
        p > PHASE_THRESHOLDS["active"]
        for p in [tensor_pct, vector_pct, scalar_pct, gpsimd_pct]
    )
    dma_active = dma_pct > PHASE_THRESHOLDS["active"]
    all_low = not compute_active and not dma_active

    if all_low:
        return "IDLE"
    if dma_active and not compute_active:
        # Distinguish LOAD vs STORE by context (simplified: use DMA direction)
        return "LOAD"  # We refine this per-window if DMA direction info available
    if compute_active and not dma_active:
        return "COMPUTE"
    if compute_active and dma_active:
        return "MIXED"
    return "IDLE"


def compute_time_windows(data, total_time_ns):
    """Compute multi-scale time window analysis."""
    active_times = data.get("active_time", [])
    dma_events = data.get("dma", [])

    if total_time_ns <= 0:
        return {}

    # Build interval lists per engine
    engine_intervals = defaultdict(list)
    for at in active_times:
        eng = normalize_engine(at.get("engine", ""))
        start = at["start_ts"]
        end = at["end_ts"]
        engine_intervals[eng].append((start, end))

    dma_intervals = []
    for d in dma_events:
        ts = d["timestamp"]
        dur = d.get("duration", 0)
        dma_intervals.append((ts, ts + dur))

    scales = {"scale_100": 100, "scale_50": 50, "scale_10": 10}
    result = {}

    for scale_name, num_windows in scales.items():
        window_size = total_time_ns / num_windows
        windows = []

        for i in range(num_windows):
            w_start = i * window_size
            w_end = (i + 1) * window_size

            pcts = {}
            for eng in ["tensor", "vector", "scalar", "gpsimd"]:
                overlap = sum(
                    intervals_overlap(w_start, w_end, s, e)
                    for s, e in engine_intervals.get(eng, [])
                )
                pcts[eng] = (
                    round(overlap / window_size * 100, 2) if window_size > 0 else 0
                )

            dma_overlap = sum(
                intervals_overlap(w_start, w_end, s, e) for s, e in dma_intervals
            )
            dma_pct = (
                round(dma_overlap / window_size * 100, 2) if window_size > 0 else 0
            )

            phase = classify_phase(
                pcts["tensor"], pcts["vector"], pcts["scalar"], pcts["gpsimd"], dma_pct
            )

            windows.append(
                {
                    "window_idx": i,
                    "start_ns": round(w_start, 2),
                    "end_ns": round(w_end, 2),
                    "tensor_pct": pcts["tensor"],
                    "vector_pct": pcts["vector"],
                    "scalar_pct": pcts["scalar"],
                    "gpsimd_pct": pcts["gpsimd"],
                    "dma_pct": dma_pct,
                    "phase": phase,
                }
            )

        # Merge adjacent windows with same phase into phase segments
        phases = []
        if windows:
            current = {
                "phase": windows[0]["phase"],
                "start_ns": windows[0]["start_ns"],
                "end_ns": windows[0]["end_ns"],
                "dominant_engines": [],
            }
            for w in windows[1:]:
                if w["phase"] == current["phase"]:
                    current["end_ns"] = w["end_ns"]
                else:
                    current["duration_ns"] = round(
                        current["end_ns"] - current["start_ns"], 2
                    )
                    # Determine dominant engines for this phase
                    phase_windows = [
                        ww
                        for ww in windows
                        if ww["start_ns"] >= current["start_ns"]
                        and ww["end_ns"] <= current["end_ns"]
                    ]
                    dominant = []
                    for eng in ["tensor", "vector", "scalar", "gpsimd", "dma"]:
                        key = f"{eng}_pct"
                        avg = (
                            mean(pw[key] for pw in phase_windows)
                            if phase_windows
                            else 0
                        )
                        if avg > PHASE_THRESHOLDS["active"]:
                            dominant.append(eng)
                    current["dominant_engines"] = dominant
                    phases.append(current)
                    current = {
                        "phase": w["phase"],
                        "start_ns": w["start_ns"],
                        "end_ns": w["end_ns"],
                        "dominant_engines": [],
                    }
            # Finalize last phase
            current["duration_ns"] = round(current["end_ns"] - current["start_ns"], 2)
            phase_windows = [
                ww
                for ww in windows
                if ww["start_ns"] >= current["start_ns"]
                and ww["end_ns"] <= current["end_ns"]
            ]
            dominant = []
            for eng in ["tensor", "vector", "scalar", "gpsimd", "dma"]:
                key = f"{eng}_pct"
                avg = mean(pw[key] for pw in phase_windows) if phase_windows else 0
                if avg > PHASE_THRESHOLDS["active"]:
                    dominant.append(eng)
            current["dominant_engines"] = dominant
            phases.append(current)

        result[scale_name] = {
            "window_size_ns": round(window_size, 2),
            "windows": windows,
            "phases": phases,
        }

    return result


# ---------------------------------------------------------------------------
# Dimension 3: Cross-Engine Blocking
# ---------------------------------------------------------------------------


def analyze_blocking(data):
    """Find cross-engine blocking patterns from instructions with evt_wait_time > 0."""
    instructions = data.get("instruction", [])
    active_times = data.get("active_time", [])
    dma_events = data.get("dma", [])

    # Build sorted interval lists
    engine_intervals = defaultdict(list)
    for at in active_times:
        eng = normalize_engine(at.get("engine", ""))
        engine_intervals[eng].append((at["start_ts"], at["end_ts"]))

    dma_intervals = [
        (d["timestamp"], d["timestamp"] + d.get("duration", 0)) for d in dma_events
    ]

    blocking_events = []
    for inst in instructions:
        wait = inst.get("evt_wait_time", 0)
        if wait <= 0:
            continue

        eng = normalize_engine(inst.get("subgroup", ""))
        ts = inst["timestamp"]
        wait_start = ts - wait
        wait_end = ts

        concurrent = []
        # Check other compute engines
        for other_eng, intervals in engine_intervals.items():
            if other_eng == eng:
                continue
            for s, e in intervals:
                overlap = intervals_overlap(wait_start, wait_end, s, e)
                if overlap > 0:
                    concurrent.append(
                        {
                            "engine": other_eng,
                            "type": "compute",
                            "overlap_ns": overlap,
                            "overlap_pct": (
                                round(overlap / wait * 100, 2) if wait > 0 else 0
                            ),
                        }
                    )

        # Check DMA
        for s, e in dma_intervals:
            overlap = intervals_overlap(wait_start, wait_end, s, e)
            if overlap > 0:
                concurrent.append(
                    {
                        "engine": "dma",
                        "type": "dma",
                        "overlap_ns": overlap,
                        "overlap_pct": (
                            round(overlap / wait * 100, 2) if wait > 0 else 0
                        ),
                    }
                )

        # Classify pattern
        if concurrent:
            has_compute = any(c["type"] == "compute" for c in concurrent)
            has_dma = any(c["type"] == "dma" for c in concurrent)
            if has_dma and not has_compute:
                pattern = "DATA_DEPENDENCY"
            elif has_compute and not has_dma:
                pattern = "ENGINE_DEPENDENCY"
            else:
                pattern = "SEMAPHORE_STALL"
        else:
            pattern = "SEMAPHORE_STALL"

        blocking_events.append(
            {
                "engine": eng,
                "opcode": inst.get("opcode", ""),
                "wait_ns": wait,
                "timestamp": ts,
                "concurrent_activities": concurrent,
                "pattern": pattern,
            }
        )

    # Sort by wait_ns descending
    blocking_events.sort(key=lambda x: x["wait_ns"], reverse=True)
    return blocking_events


# ---------------------------------------------------------------------------
# Dimension 7: DMA-Compute Overlap Ratio
# ---------------------------------------------------------------------------


def analyze_dma_compute_overlap(data):
    """Calculate DMA-compute overlap ratio."""
    active_times = data.get("active_time", [])
    dma_events = data.get("dma", [])

    # Compute engine intervals (merged)
    compute_intervals = []
    for at in active_times:
        eng = normalize_engine(at.get("engine", ""))
        if eng in ("tensor", "vector", "scalar", "gpsimd"):
            compute_intervals.append((at["start_ts"], at["end_ts"]))

    # DMA intervals
    dma_intervals = []
    total_dma_ns = 0
    for d in dma_events:
        ts = d["timestamp"]
        dur = d.get("duration", 0)
        dma_intervals.append((ts, ts + dur))
        total_dma_ns += dur

    # Calculate overlap: for each DMA interval, check overlap with any compute interval
    overlapped_ns = 0
    for ds, de in dma_intervals:
        for cs, ce in compute_intervals:
            overlapped_ns += intervals_overlap(ds, de, cs, ce)

    # Cap overlap at total DMA time (overlaps with multiple compute engines count once)
    # Actually we should merge compute intervals first to avoid double counting
    merged_compute = merge_intervals(compute_intervals)
    overlapped_ns = 0
    for ds, de in dma_intervals:
        for cs, ce in merged_compute:
            overlapped_ns += intervals_overlap(ds, de, cs, ce)

    overlap_ratio = overlapped_ns / total_dma_ns if total_dma_ns > 0 else 0

    return {
        "total_dma_ns": total_dma_ns,
        "dma_overlapped_with_compute_ns": overlapped_ns,
        "overlap_ratio": round(overlap_ratio, 4),
        "non_overlapped_dma_ns": total_dma_ns - overlapped_ns,
    }


def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for s, e in sorted_intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# Dimension 11: Wait Chain / Semaphore Trace
# ---------------------------------------------------------------------------


def analyze_wait_chains(data):
    """Trace wait chains: who waited how long for whom."""
    instructions = data.get("instruction", [])
    sem_updates = data.get("semaphore_update", [])
    dma_events = data.get("dma", [])

    # Build semaphore update index: group by id, sorted by timestamp
    sem_by_id = defaultdict(list)
    for su in sem_updates:
        sem_by_id[su["id"]].append(su)
    for sid in sem_by_id:
        sem_by_id[sid].sort(key=lambda x: x["timestamp"])

    # Build instruction index by engine for blocker lookup
    insts_by_engine = defaultdict(list)
    for inst in instructions:
        eng = normalize_engine(inst.get("subgroup", ""))
        insts_by_engine[eng].append(inst)
    for eng in insts_by_engine:
        insts_by_engine[eng].sort(key=lambda x: x["timestamp"])

    # Build DMA index with semaphore_id for matching
    dma_by_sem = defaultdict(list)
    for d in dma_events:
        sem_id = d.get("semaphore_id", "")
        if sem_id:
            # Extract numeric semaphore ID: "S[24] (qSyncIO0)" -> "24"
            import re

            m = re.search(r"S\[(\d+)\]", sem_id)
            if m:
                dma_by_sem[m.group(1)].append(d)

    wait_chains = []
    for inst in instructions:
        wait = inst.get("evt_wait_time", 0)
        if wait <= 0:
            continue

        waiter_eng = normalize_engine(inst.get("subgroup", ""))
        waiter_ts = inst["timestamp"]
        wait_start = waiter_ts - wait

        # Find the semaphore update closest to and before the waiter's timestamp
        # We check all semaphore IDs since we don't know which one this instruction waited on
        best_match = None
        best_gap = float("inf")

        for sid, updates in sem_by_id.items():
            for su in reversed(updates):
                if su["timestamp"] <= waiter_ts and su["timestamp"] >= wait_start:
                    gap = waiter_ts - su["timestamp"]
                    if gap < best_gap:
                        best_gap = gap
                        best_match = {
                            "sem_id": sid,
                            "sem_ts": su["timestamp"],
                            "sem_value": su["value"],
                        }
                    break  # Only need the latest one per sem_id

        if not best_match:
            continue

        # Find which instruction or DMA triggered this semaphore release
        blocker = None
        sem_ts = best_match["sem_ts"]
        sem_id = best_match["sem_id"]

        # Check DMA events that use this semaphore
        if sem_id in dma_by_sem:
            for d in dma_by_sem[sem_id]:
                dma_end = d["timestamp"] + d.get("duration", 0)
                if abs(dma_end - sem_ts) < 100:  # Within 100 ticks
                    blocker = {
                        "engine": "dma",
                        "opcode": d.get("op", "dma"),
                        "timestamp": d["timestamp"],
                        "semaphore_id": sem_id,
                        "variable": d.get("variable", ""),
                    }
                    break

        # If no DMA match, check compute instructions
        if not blocker:
            for eng, insts in insts_by_engine.items():
                if eng == waiter_eng:
                    continue
                for other_inst in insts:
                    other_end = other_inst["timestamp"] + other_inst.get("duration", 0)
                    if abs(other_end - sem_ts) < 100:
                        blocker = {
                            "engine": eng,
                            "opcode": other_inst.get("opcode", ""),
                            "timestamp": other_inst["timestamp"],
                            "semaphore_id": sem_id,
                        }
                        break
                if blocker:
                    break

        chain_entry = {
            "waiter": {
                "engine": waiter_eng,
                "opcode": inst.get("opcode", ""),
                "timestamp": waiter_ts,
                "wait_ns": wait,
            },
            "semaphore": best_match,
            "blocker": blocker,
            "wait_duration_ns": wait,
        }
        wait_chains.append(chain_entry)

    # Sort by wait duration descending
    wait_chains.sort(key=lambda x: x["wait_duration_ns"], reverse=True)
    return wait_chains


# ---------------------------------------------------------------------------
# Dimension 12: Critical Path Analysis
# ---------------------------------------------------------------------------


def analyze_critical_path(data, wait_chains):
    """Build dependency graph from wait chains and find the critical (longest) path."""
    instructions = data.get("instruction", [])

    if not wait_chains:
        total_compute = sum(i.get("duration", 0) for i in instructions)
        return {
            "path": [],
            "total_compute_ns": total_compute,
            "total_wait_ns": 0,
            "efficiency_pct": 100.0,
        }

    # Build adjacency: blocker -> waiter edges
    # Nodes are (engine, timestamp) tuples
    nodes = {}  # node_id -> {engine, opcode, timestamp, duration, wait_ns}
    edges = []  # (from_id, to_id, weight)

    for inst in instructions:
        eng = normalize_engine(inst.get("subgroup", ""))
        ts = inst["timestamp"]
        node_id = f"{eng}_{ts}"
        nodes[node_id] = {
            "engine": eng,
            "opcode": inst.get("opcode", ""),
            "timestamp": ts,
            "duration": inst.get("duration", 0),
            "wait_ns": inst.get("evt_wait_time", 0),
        }

    for wc in wait_chains:
        if not wc.get("blocker"):
            continue
        b = wc["blocker"]
        w = wc["waiter"]
        b_id = f"{b['engine']}_{b['timestamp']}"
        w_id = f"{w['engine']}_{w['timestamp']}"
        if b_id in nodes and w_id in nodes:
            edges.append((b_id, w_id, wc["wait_duration_ns"]))

    # Find longest path using topological sort (DAG)
    # Build adjacency list
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set(nodes.keys())
    for frm, to, weight in edges:
        adj[frm].append((to, weight))
        in_degree[to] += 1
        all_nodes.add(frm)
        all_nodes.add(to)

    # Topological sort by timestamp
    sorted_nodes = sorted(
        all_nodes, key=lambda n: nodes[n]["timestamp"] if n in nodes else 0
    )

    # Longest path via DP
    dist = {n: 0 for n in all_nodes}
    parent = {n: None for n in all_nodes}

    for n in sorted_nodes:
        for neighbor, weight in adj.get(n, []):
            node_duration = nodes[n]["duration"] if n in nodes else 0
            new_dist = dist[n] + node_duration + weight
            if new_dist > dist.get(neighbor, 0):
                dist[neighbor] = new_dist
                parent[neighbor] = n

    # Find the node with maximum distance
    if not dist:
        return {
            "path": [],
            "total_compute_ns": 0,
            "total_wait_ns": 0,
            "efficiency_pct": 100.0,
        }

    end_node = max(dist, key=dist.get)

    # Trace back the path
    path = []
    current = end_node
    while current is not None:
        if current in nodes:
            path.append(nodes[current])
        current = parent.get(current)
    path.reverse()

    total_compute = sum(n.get("duration", 0) for n in path)
    total_wait = sum(n.get("wait_ns", 0) for n in path)
    total = total_compute + total_wait
    efficiency = round(total_compute / total * 100, 2) if total > 0 else 100.0

    # Add cumulative time to path nodes
    cumulative = 0
    for node in path:
        cumulative += node.get("duration", 0) + node.get("wait_ns", 0)
        node["cumulative_ns"] = cumulative

    return {
        "path": path,
        "path_length": len(path),
        "total_compute_ns": total_compute,
        "total_wait_ns": total_wait,
        "critical_path_ns": total,
        "efficiency_pct": efficiency,
    }


# ---------------------------------------------------------------------------
# Dimension 15: Anomaly Detection
# ---------------------------------------------------------------------------


def detect_anomalies(data, threshold_ratio=3.0):
    """Detect instructions with anomalous durations (> threshold_ratio * mean)."""
    instructions = data.get("instruction", [])

    # Group by engine
    by_engine = defaultdict(list)
    for inst in instructions:
        eng = normalize_engine(inst.get("subgroup", ""))
        by_engine[eng].append(inst)

    anomalies = []
    for eng, insts in by_engine.items():
        durations = [i.get("duration", 0) for i in insts]
        if len(durations) < 3:
            continue
        mu = mean(durations)
        if mu == 0:
            continue
        sd = stdev(durations) if len(durations) > 1 else 0

        for inst in insts:
            dur = inst.get("duration", 0)
            ratio = dur / mu if mu > 0 else 0
            if ratio > threshold_ratio:
                anomalies.append(
                    {
                        "engine": eng,
                        "opcode": inst.get("opcode", ""),
                        "duration_ns": dur,
                        "mean_ns": round(mu, 2),
                        "stddev_ns": round(sd, 2),
                        "ratio": round(ratio, 2),
                        "timestamp": inst.get("timestamp", 0),
                        "layer": inst.get("layer", ""),
                        "hlo_name": inst.get("hlo_name", ""),
                        "operands": inst.get("operands", ""),
                        "evt_wait_time": inst.get("evt_wait_time", 0),
                    }
                )

    # Sort by ratio descending
    anomalies.sort(key=lambda x: x["ratio"], reverse=True)
    return anomalies


# ---------------------------------------------------------------------------
# Dimension 16: Repetition Pattern Detection
# ---------------------------------------------------------------------------


def detect_repetition_patterns(data, min_pattern_len=3, min_count=2):
    """Detect repeated opcode subsequences per engine."""
    instructions = data.get("instruction", [])

    by_engine = defaultdict(list)
    for inst in instructions:
        eng = normalize_engine(inst.get("subgroup", ""))
        by_engine[eng].append(inst)

    patterns = []
    for eng, insts in by_engine.items():
        # Sort by timestamp
        insts_sorted = sorted(insts, key=lambda x: x["timestamp"])
        opcodes = [i.get("opcode", "") for i in insts_sorted]
        timestamps = [i["timestamp"] for i in insts_sorted]
        durations = [i.get("duration", 0) for i in insts_sorted]

        if len(opcodes) < min_pattern_len * min_count:
            continue

        # Sliding window to find repeated subsequences
        found = {}
        for plen in range(min_pattern_len, min(len(opcodes) // min_count + 1, 10)):
            for start in range(len(opcodes) - plen + 1):
                subseq = tuple(opcodes[start : start + plen])
                if subseq not in found:
                    found[subseq] = []
                found[subseq].append(start)

        for subseq, positions in found.items():
            if len(positions) < min_count:
                continue
            # Check that positions don't overlap
            non_overlapping = [positions[0]]
            for p in positions[1:]:
                if p >= non_overlapping[-1] + len(subseq):
                    non_overlapping.append(p)
            if len(non_overlapping) < min_count:
                continue

            # Calculate avg duration per iteration
            iter_durations = []
            for p in non_overlapping:
                iter_dur = sum(durations[p : p + len(subseq)])
                iter_durations.append(iter_dur)

            # Calculate intervals between iterations
            intervals = []
            for i in range(1, len(non_overlapping)):
                gap = (
                    timestamps[non_overlapping[i]] - timestamps[non_overlapping[i - 1]]
                )
                intervals.append(gap)

            patterns.append(
                {
                    "engine": eng,
                    "pattern": list(subseq),
                    "count": len(non_overlapping),
                    "avg_duration_per_iteration_ns": (
                        round(mean(iter_durations), 2) if iter_durations else 0
                    ),
                    "avg_interval_ns": round(mean(intervals), 2) if intervals else 0,
                }
            )

    # Deduplicate: keep longest patterns that subsume shorter ones
    patterns.sort(key=lambda x: len(x["pattern"]), reverse=True)
    filtered = []
    seen_engines = defaultdict(set)
    for p in patterns:
        key = (p["engine"], tuple(p["pattern"]))
        # Check if this is a subsequence of an already-kept pattern
        skip = False
        for kept in filtered:
            if kept["engine"] == p["engine"] and len(kept["pattern"]) > len(
                p["pattern"]
            ):
                # Check subsequence
                kept_str = " ".join(kept["pattern"])
                p_str = " ".join(p["pattern"])
                if p_str in kept_str and kept["count"] >= p["count"]:
                    skip = True
                    break
        if not skip:
            filtered.append(p)

    return filtered


# ---------------------------------------------------------------------------
# Dimension 17: Engine Handoff Graph
# ---------------------------------------------------------------------------


def analyze_engine_handoffs(data, gap_threshold_ns=200):
    """Detect engine handoff patterns from active_time intervals."""
    active_times = data.get("active_time", [])

    # Sort all intervals by end time
    intervals = []
    for at in active_times:
        eng = normalize_engine(at.get("engine", ""))
        intervals.append(
            {
                "engine": eng,
                "start": at["start_ts"],
                "end": at["end_ts"],
            }
        )

    # Also sort by start time for lookup
    by_start = sorted(intervals, key=lambda x: x["start"])

    # For each interval end, find which engine starts within gap_threshold_ns
    handoff_counts = defaultdict(lambda: {"count": 0, "gaps": []})

    for interval in intervals:
        end_ts = interval["end"]
        from_eng = interval["engine"]

        # Binary-ish search: find intervals starting within [end_ts, end_ts + gap_threshold_ns]
        for other in by_start:
            if other["start"] < end_ts:
                continue
            if other["start"] > end_ts + gap_threshold_ns:
                break
            if other["engine"] == from_eng:
                continue
            key = (from_eng, other["engine"])
            handoff_counts[key]["count"] += 1
            handoff_counts[key]["gaps"].append(other["start"] - end_ts)

    handoffs = []
    for (from_eng, to_eng), data_val in handoff_counts.items():
        avg_gap = mean(data_val["gaps"]) if data_val["gaps"] else 0
        handoffs.append(
            {
                "from_engine": from_eng,
                "to_engine": to_eng,
                "count": data_val["count"],
                "avg_gap_ns": round(avg_gap, 2),
            }
        )

    # Sort by count descending
    handoffs.sort(key=lambda x: x["count"], reverse=True)
    return handoffs


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def find_neff_files(report_dir):
    """Find neff_*_full.json files, skipping system_*.json."""
    files = []
    for f in os.listdir(report_dir):
        if f.startswith("neff_") and f.endswith("_full.json"):
            files.append(os.path.join(report_dir, f))
    return sorted(files)


def analyze_single_neff(filepath):
    """Run full analysis pipeline on a single neff JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    neff_name = os.path.basename(filepath)
    meta = data["metadata"][0] if data.get("metadata") else {}

    # Get total time from metadata timestamps
    first_ts = meta.get("first_hw_timestamp", 0)
    last_ts = meta.get("last_hw_timestamp", 0)
    total_time_ns = last_ts - first_ts
    if total_time_ns <= 0:
        # Fallback to summary
        s = data.get("summary", [{}])[0]
        total_time_ns = s.get("total_time", 0) * 1e9

    # Run all analyses
    summary = extract_summary(data)
    time_windows = compute_time_windows(data, total_time_ns)
    blocking = analyze_blocking(data)
    dma_overlap = analyze_dma_compute_overlap(data)
    wait_chains = analyze_wait_chains(data)
    critical_path = analyze_critical_path(data, wait_chains)
    anomalies = detect_anomalies(data)
    repetitions = detect_repetition_patterns(data)
    handoffs = analyze_engine_handoffs(data)

    return {
        "neff_name": neff_name,
        "summary": summary,
        "time_windows": time_windows,
        "blocking": blocking,
        "dma_compute_overlap": dma_overlap,
        "wait_chains": wait_chains,
        "critical_path": critical_path,
        "anomalies": anomalies,
        "repetition_patterns": repetitions,
        "engine_handoffs": handoffs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Neuron Profiler JSON Feature Engineering Tool"
    )
    parser.add_argument(
        "report_dir",
        help="Path to the Neuron profiler report directory containing neff_*_full.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file path (default: stdout)",
        default=None,
    )
    args = parser.parse_args()

    if not os.path.isdir(args.report_dir):
        print(f"Error: {args.report_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    neff_files = find_neff_files(args.report_dir)
    if not neff_files:
        print(
            f"Error: No neff_*_full.json files found in {args.report_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    results = []
    for fp in neff_files:
        print(f"Analyzing {os.path.basename(fp)}...", file=sys.stderr)
        result = analyze_single_neff(fp)
        results.append(result)

    # If single file, output directly; if multiple, wrap in array
    output = results[0] if len(results) == 1 else results

    output_json = json.dumps(output, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
