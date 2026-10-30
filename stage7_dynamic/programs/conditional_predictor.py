from dataclasses import dataclass, field
import math
import statistics


VERSION = "conditional-graph-critical-path-v2"


def scale(value):
    return max(16.0, 0.25 * value)


def quantile(values, probability=0.9):
    if not values:
        raise ValueError("empty empirical distribution")
    return sorted(values)[max(0, math.ceil(probability * len(values)) - 1)]


def safe_prefill(profile, new, cached=0):
    try:
        return profile.prefill(new, cached)
    except ValueError:
        limit = 2050
        unit = profile.prefill(limit, 0) / limit
        return max(1e-5, unit * max(0, new))


def _progress_values(member_progress):
    if member_progress is None:
        return None
    if isinstance(member_progress, dict):
        return [max(0, int(value)) for value in member_progress.values()]
    values = tuple(member_progress)
    if values and isinstance(values[0], (tuple, list)) and len(values[0]) == 2:
        values = [value for _, value in values]
    return [max(0, int(value)) for value in values]


def _member_outputs(record):
    if "member_outputs" in record:
        return tuple(float(value) for value in record["member_outputs"])
    members = record.get("members")
    if members:
        return tuple(float(member["outputs"]) for member in members)
    return None


@dataclass(frozen=True)
class ConditionalPredictor:
    records: tuple
    away_mean: float
    away_q: float
    q_cal: float
    q_regions: tuple
    neighbors: int = 32
    version: str = VERSION
    risk_q: float = 0.0
    _neighbor_cache: object = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self):
        object.__setattr__(self, "_neighbor_cache", {})

    def _neighbors(self, prompt, ordinal, slots):
        key = (prompt, ordinal, slots)
        if key in self._neighbor_cache:
            return self._neighbor_cache[key]

        def distance(record):
            return (abs(math.log1p(prompt) - math.log1p(record["prompt"]))
                    + 0.5 * abs(ordinal - record["ordinal"])
                    + 0.25 * abs(slots - record.get("slots", 1)), record["graph_id"])

        result = sorted(self.records, key=distance)[:self.neighbors]
        self._neighbor_cache[key] = result
        return result

    @staticmethod
    def _exact(records, prompt, ordinal, slots):
        return [record for record in records
                if record["prompt"] == prompt and record["ordinal"] == ordinal
                and record.get("slots", 1) == slots]

    @staticmethod
    def _mode(neighbors, exact, prompt, ordinal, slots):
        if exact:
            return "exact"
        aligned = [record for record in neighbors
                   if record["ordinal"] == ordinal and record.get("slots", 1) == slots]
        if not aligned:
            return "extrapolated"
        prompts = [record["prompt"] for record in aligned]
        return "backoff" if min(prompts) <= prompt <= max(prompts) else "extrapolated"

    @staticmethod
    def _targets(record, slots):
        observed = _member_outputs(record)
        if observed is not None and len(observed) == slots:
            return observed
        return tuple(record["outputs"] / max(1, record.get("slots", 1))
                     for _ in range(slots))

    def _future_details(self, selected, ordinal, profile, mode):
        aligned = [record for record in selected if record["ordinal"] == ordinal]
        donors = aligned or list(selected)
        details = []
        max_offset = max((len(record["future"]) for record in donors), default=0)
        for offset in range(max_offset):
            futures = [(record["graph_id"], record["future"][offset])
                       for record in donors if offset < len(record["future"])]
            if not futures:
                continue
            items = [item for _, item in futures]
            prompt = statistics.fmean(item["prompt"] for item in items)
            outputs = statistics.fmean(item["outputs"] for item in items)
            width = max(1, int(round(statistics.fmean(item.get("slots", 1) for item in items))))
            compute = statistics.fmean(max(
                0.01, max(self._targets(item, item.get("slots", 1))) - 1
                + safe_prefill(profile, item["prompt"]) / profile.decode_seconds)
                for item in items)
            details.append({
                "ordinal": ordinal + offset + 1,
                "width": width,
                "prompt_tokens": prompt,
                "output_tokens": outputs,
                "compute": compute,
                "away": self.away_mean,
                "prediction_mode": ("exact" if mode == "exact" and len(
                    {graph_id for graph_id, _ in futures}) == 1 else mode),
                "graph_ids": tuple(sorted({graph_id for graph_id, _ in futures})),
            })
        return tuple(details)

    def _future(self, neighbors, ordinal, profile, slots=1):
        mode = "backoff" if neighbors else "extrapolated"
        details = self._future_details(neighbors, ordinal, profile, mode)
        return tuple(value for detail in details
                     for value in (detail["away"], detail["compute"]))

    def ready_parts(self, prompt, ordinal, slots, emitted, profile, member_progress=None):
        slots = max(1, int(slots))
        neighbors = self._neighbors(prompt, ordinal, slots)
        exact = self._exact(neighbors, prompt, ordinal, slots)
        selected = exact or neighbors
        if not selected:
            raise ValueError("no predictor support")
        mode = self._mode(neighbors, exact, prompt, ordinal, slots)
        progress = _progress_values(member_progress)
        if progress is not None and len(progress) != slots:
            raise ValueError("compound member progress width mismatch")
        mean_emitted = emitted / slots if progress is None else 0.0
        survivors = []
        for record in selected:
            targets = self._targets(record, slots)
            observed = progress if progress is not None else [mean_emitted] * slots
            if any(target > value for target, value in zip(targets, observed)):
                survivors.append(record)
        chosen = survivors or selected
        residuals = []
        for record in chosen:
            targets = self._targets(record, slots)
            observed = progress if progress is not None else [mean_emitted] * slots
            residuals.append(max(0.01, max(target - value
                                           for target, value in zip(targets, observed))))
        current = max(0.01, statistics.fmean(residuals))
        if not survivors and emitted:
            mode = "extrapolated"
        details = self._future_details(selected, ordinal, profile, mode)
        tail = tuple(value for detail in details
                     for value in (detail["away"], detail["compute"]))
        total = current + math.fsum(tail)
        return current, max(0.01, total), tail, len(survivors), mode

    def ready_base(self, prompt, ordinal, emitted, slots=1):
        class Profile:
            decode_seconds = 1.0

            @staticmethod
            def prefill(new, cached=0):
                return 0.0

        current, total, tail, survivors, _ = self.ready_parts(
            prompt, ordinal, slots, emitted, Profile)
        return current, total, tail, survivors

    def away_base(self, prompt, ordinal, outputs, elapsed_units, slots=1):
        class Profile:
            decode_seconds = 1.0

            @staticmethod
            def prefill(new, cached=0):
                return 0.0

        residual = max(0.01, self.away_mean - elapsed_units)
        neighbors = self._neighbors(prompt, ordinal, slots)
        exact = self._exact(neighbors, prompt, ordinal, slots)
        selected = exact or neighbors
        if not selected:
            raise ValueError("no predictor support")
        mode = self._mode(neighbors, exact, prompt, ordinal, slots)
        details = self._future_details(selected, ordinal, Profile, mode)
        tail = [residual]
        if details:
            tail.append(details[0]["compute"])
            for detail in details[1:]:
                tail.extend((detail["away"], detail["compute"]))
        else:
            tail.append(0.01)
        return tuple(tail), len(selected), 0, self.away_mean

    def estimate(self, goal, profile, conformal=True, slots=1, member_progress=None):
        decode_seconds = profile.decode_seconds
        slots = max(1, int(slots))
        if goal.ready is not None:
            call = goal.ready
            neighbors = self._neighbors(call.prompt_tokens, call.ordinal, slots)
            exact = self._exact(neighbors, call.prompt_tokens, call.ordinal, slots)
            selected = exact or neighbors
            current, total_mean, tail, survivors, mode = self.ready_parts(
                call.prompt_tokens, call.ordinal, slots, call.emitted, profile,
                member_progress=member_progress)
            details = self._future_details(selected, call.ordinal, profile, mode)
            region = call.ordinal
            support = {"phase": "READY", "mode": mode, "survivors": survivors,
                       "emitted": call.emitted,
                       "aggregation": "member_critical_path_max",
                       "member_progress": ("observed" if member_progress is not None
                                           else "aggregate_balanced_fallback")}
            point_current = current * decode_seconds
            current_meta = {"ordinal": call.ordinal, "width": slots,
                            "prompt_tokens": call.prompt_tokens,
                            "prediction_mode": mode}
        else:
            last = goal.history[-1]
            elapsed = goal.tool_elapsed / decode_seconds
            tail, survivors, interval, hi = self.away_base(
                last.prompt_tokens, last.ordinal, last.outputs, elapsed, slots=slots)
            tail = list(tail)
            neighbors = self._neighbors(last.prompt_tokens, last.ordinal, slots)
            exact = self._exact(neighbors, last.prompt_tokens, last.ordinal, slots)
            selected = exact or neighbors
            current_mode = self._mode(neighbors, exact, last.prompt_tokens,
                                      last.ordinal, slots)
            details = self._future_details(selected, last.ordinal, profile, current_mode)
            current = 0.0
            total_mean = math.fsum(tail)
            region = last.ordinal + 1
            mode = "survival_backoff" if current_mode != "extrapolated" else "extrapolated"
            support = {"phase": "AWAY", "mode": mode, "elapsed_units": elapsed,
                       "survival_interval": interval, "max_tool_units": hi,
                       "aggregation": "member_critical_path_max",
                       "future_modes": [detail["prediction_mode"] for detail in details]}
            point_current = 0.0
            current_meta = None
        q = dict(self.q_regions).get(region, self.q_cal) if conformal else 0.0
        current_padded = current + q * scale(current) if current else 0.0
        away_units = math.fsum(tail[::2])
        total = total_mean + q * scale(total_mean) + (
            self.away_q * scale(away_units) if conformal and away_units else 0.0)
        phases = []
        metadata = []
        if goal.ready is not None:
            phases.append(["compute", current_padded])
            metadata.append(dict(current_meta, kind="compute", prefill_tokens=0))
            extra = max(0.0, total - current_padded - math.fsum(tail))
        else:
            extra = max(0.0, total - math.fsum(tail))
        future_compute = math.fsum(tail[1::2])
        for index in range(0, len(tail), 2):
            detail = details[index // 2] if index // 2 < len(details) else {
                "ordinal": region, "width": slots, "prompt_tokens": 0.0,
                "output_tokens": 0.0, "prediction_mode": "extrapolated",
            }
            phases.append(["away", tail[index]])
            metadata.append({"kind": "away", "ordinal": detail["ordinal"],
                             "width": detail["width"], "prompt_tokens": 0,
                             "prediction_mode": detail["prediction_mode"]})
            phases.append(["compute", tail[index + 1] + (
                extra * tail[index + 1] / future_compute if future_compute else 0.0)])
            metadata.append({"kind": "compute", "ordinal": detail["ordinal"],
                             "width": detail["width"],
                             "prompt_tokens": detail["prompt_tokens"],
                             "prefill_tokens": detail["prompt_tokens"],
                             "output_tokens": detail.get("output_tokens", 0.0),
                             "prediction_mode": detail["prediction_mode"]})
        paired = [(phase, meta) for phase, meta in zip(phases, metadata) if phase[1] > 0]
        phases = [(kind, max(0.0, value) * decode_seconds)
                  for (kind, value), _ in paired]
        metadata = [meta for _, meta in paired]
        if phases:
            index = max(i for i, (kind, _) in enumerate(phases) if kind == "compute")
            correction = total * decode_seconds - math.fsum(
                value for i, (_, value) in enumerate(phases) if i != index)
            if correction < -1e-9:
                raise RuntimeError("phase budget negative")
            phases[index] = ("compute", max(0.0, correction))
        if not math.isclose(math.fsum(value for _, value in phases),
                            total * decode_seconds, abs_tol=1e-8):
            raise RuntimeError("phase conservation")
        return {
            "point": total_mean * decode_seconds,
            "work": total * decode_seconds,
            "current": current_padded * decode_seconds,
            "current_point": point_current,
            "mean_compute": (total_mean - away_units) * decode_seconds,
            "mean_away": away_units * decode_seconds,
            "phases": tuple(phases),
            "phase_metadata": tuple(metadata),
            "support": support,
            "prediction_mode": mode,
            "conformal": conformal,
            "q_region": region,
        }

    def to_dict(self):
        return {"records": self.records, "away_mean": self.away_mean,
                "away_q": self.away_q, "q_cal": self.q_cal,
                "q_regions": self.q_regions, "neighbors": self.neighbors,
                "version": self.version, "risk_q": self.risk_q}

    @classmethod
    def from_dict(cls, value):
        records = tuple((dict(record) for record in value["records"]))
        for record in records:
            record["future"] = tuple(dict(item) for item in record["future"])
        return cls(records, value["away_mean"], value["away_q"], value["q_cal"],
                   tuple((int(key), item) for key, item in value["q_regions"]),
                   value["neighbors"], value["version"], value.get("risk_q", 0.0))
