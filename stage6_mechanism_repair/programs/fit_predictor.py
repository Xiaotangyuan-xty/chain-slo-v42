import hashlib
import json
import math
import statistics

from conditional_predictor import ConditionalPredictor, safe_prefill, scale, quantile


def graph_signature(stages):
    result = []
    for stage in stages:
        requests = stage["requests"] if isinstance(stage, dict) else stage
        result.append(sorted((request["input_tokens"], request["output_tokens"])
                             for request in requests))
    return json.dumps(result, separators=(",", ":"))


def split_graph_ids(graphs):
    ids = sorted((graph["request_id"] for graph in graphs),
                 key=lambda graph_id: hashlib.sha256(f"repair-v1/{graph_id}".encode()).digest())
    if len(ids) != 394 or len(set(ids)) != 394:
        raise ValueError("unexpected training graph identity")
    return set(ids[:295]), set(ids[295:])


def make_records(graphs, selected_ids):
    records = []
    for graph in graphs:
        if graph["request_id"] not in selected_ids:
            continue
        stages = graph["stages"]
        for ordinal, stage in enumerate(stages):
            requests = stage["requests"]
            future = tuple({
                "prompt": sum(item["input_tokens"] for item in later["requests"]),
                "outputs": sum(item["output_tokens"] for item in later["requests"]),
                "slots": len(later["requests"]),
            } for later in stages[ordinal + 1:])
            records.append({
                "graph_id": graph["request_id"],
                "ordinal": ordinal,
                "prompt": sum(item["input_tokens"] for item in requests),
                "outputs": sum(item["output_tokens"] for item in requests),
                "slots": len(requests),
                "future": future,
            })
    return tuple(records)


def away_parameters(old_predictor):
    durations = []
    for _, records in old_predictor["groups"]:
        for record in records:
            tail = record[4]
            if tail:
                durations.append(tail[0])
    if not durations:
        raise ValueError("no registered AWAY parameters")
    return statistics.fmean(durations), old_predictor["q_cal"]


def fit(records, away_mean, away_q):
    if not records:
        raise ValueError("empty FIT records")
    return ConditionalPredictor(records, away_mean, away_q, 0.0, ())


def calibration_points(outputs):
    step = max(1, outputs // 16)
    return sorted(set(range(0, outputs, step)) | {max(0, outputs - 1)})


def actual_tail(record, away_mean, profile):
    tail = []
    for future in record["future"]:
        compute = max(0.01, future["outputs"] - 1
                      + safe_prefill(profile, future["prompt"]) / profile.decode_seconds)
        tail.extend((away_mean, compute))
    return tail


def calibrate(base, cal_records, profile):
    by_graph = {}
    for record in cal_records:
        by_graph.setdefault(record["graph_id"], []).append(record)
    scores = []
    regions = {}
    for graph_id, records in by_graph.items():
        graph_score = -math.inf
        graph_regions = set()
        for record in records:
            for emitted in calibration_points(record["outputs"]):
                current, total, tail, _, _ = base.ready_parts(
                    record["prompt"], record["ordinal"], record["slots"], emitted, profile)
                truth_current = record["outputs"] - emitted
                truth_total = truth_current + math.fsum(actual_tail(record, base.away_mean, profile))
                score = max((truth_current - current) / scale(current),
                            (truth_total - total) / scale(total))
                graph_score = max(graph_score, score)
                graph_regions.add(record["ordinal"])
        scores.append(graph_score)
        for ordinal in graph_regions:
            regions.setdefault(ordinal, []).append(graph_score)
    if not scores:
        raise ValueError("empty CAL scores")
    q_global = max(0.0, quantile(scores, 0.9))
    q_regions = {ordinal: max(0.0, quantile(values, 0.9)) if len(values) >= 10 else q_global
                 for ordinal, values in regions.items()}
    calibrated = ConditionalPredictor(base.records, base.away_mean, base.away_q,
                                      q_global, tuple(sorted(q_regions.items())))
    return calibrated, {"graphs": len(scores), "q_global": q_global,
                        "q_regions": q_regions, "max_score": max(scores)}


def evaluate(model, cal_records, profile):
    total = 0
    covered = 0
    by_ordinal = {}
    for record in cal_records:
        for emitted in calibration_points(record["outputs"]):
            current, total_mean, tail, _, _ = model.ready_parts(
                record["prompt"], record["ordinal"], record["slots"], emitted, profile)
            q = dict(model.q_regions).get(record["ordinal"], model.q_cal)
            predicted_current = current + q * scale(current)
            away_units = math.fsum(tail[::2])
            predicted_total = (total_mean + q * scale(total_mean)
                               + model.away_q * scale(away_units))
            truth_current = record["outputs"] - emitted
            truth_total = truth_current + math.fsum(actual_tail(record, model.away_mean, profile))
            total += 2
            covered += int(predicted_current >= truth_current) + int(predicted_total >= truth_total)
            region = by_ordinal.setdefault(record["ordinal"], [0, 0])
            region[0] += 2
            region[1] += int(predicted_current >= truth_current) + int(predicted_total >= truth_total)
    return {
        "states": total,
        "coverage": covered / total if total else 0.0,
        "by_ordinal": {str(ordinal): {"states": values[0], "coverage": values[1] / values[0]}
                       for ordinal, values in sorted(by_ordinal.items())},
        "scope": "READY intrinsic output/prefill and registered AWAY parameters only; not JCT coverage",
    }
