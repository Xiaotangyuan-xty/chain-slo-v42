from dataclasses import dataclass, field
import math
import statistics


VERSION = "conditional-graph-backoff-v1"


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


@dataclass(frozen=True)
class ConditionalPredictor:
    records: tuple
    away_mean: float
    away_q: float
    q_cal: float
    q_regions: tuple
    neighbors: int = 32
    version: str = VERSION
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
                    + 0.25 * abs(slots - record["slots"]), record["graph_id"])
        result = sorted(self.records, key=distance)[:self.neighbors]
        self._neighbor_cache[key] = result
        return result

    def _future(self, neighbors, ordinal, profile):
        tail = []
        max_ordinal = max((len(record["future"]) for record in neighbors), default=0)
        for offset in range(max_ordinal):
            futures = [record["future"][offset] for record in neighbors
                       if offset < len(record["future"])]
            if not futures:
                continue
            prompt = statistics.fmean(item["prompt"] for item in futures)
            outputs = statistics.fmean(item["outputs"] for item in futures)
            compute = max(0.01, outputs - 1 + safe_prefill(profile, prompt) / profile.decode_seconds)
            tail.extend((self.away_mean, compute))
        return tuple(tail)

    def ready_parts(self, prompt, ordinal, slots, emitted, profile):
        neighbors = self._neighbors(prompt, ordinal, slots)
        exact = [record for record in neighbors
                 if record["prompt"] == prompt and record["ordinal"] == ordinal
                 and record["slots"] == slots]
        mode = "exact" if exact else "backoff"
        selected = exact or neighbors
        survivors = [record for record in selected if record["outputs"] > emitted]
        if survivors:
            current = statistics.fmean(record["outputs"] - emitted for record in survivors)
        else:
            current = max(1.0, statistics.fmean(record["outputs"] for record in selected) - emitted)
        current = max(0.01, current)
        tail = self._future(neighbors, ordinal, profile)
        total = current + math.fsum(tail)
        return current, max(0.01, total), tail, len(survivors), mode

    def ready_base(self, prompt, ordinal, emitted, slots=1):
        class Profile:
            decode_seconds = 1.0
            @staticmethod
            def prefill(new, cached=0):
                return 0.0
        current, total, tail, survivors, _ = self.ready_parts(prompt, ordinal, slots, emitted, Profile)
        return current, total, tail, survivors

    def away_base(self, prompt, ordinal, outputs, elapsed_units):
        class Profile:
            decode_seconds = 1.0
            @staticmethod
            def prefill(new, cached=0):
                return 0.0
        residual = max(0.01, self.away_mean - elapsed_units)
        tail = [residual]
        future = self._future(self._neighbors(prompt, ordinal + 1, 1), ordinal + 1, Profile)
        tail.extend(future)
        if not future:
            tail.append(0.01)
        return tuple(tail), len(self.records), 0, self.away_mean

    def estimate(self, goal, profile, conformal=True, slots=1):
        decode_seconds = profile.decode_seconds
        if goal.ready is not None:
            call = goal.ready
            current, total_mean, tail, survivors, mode = self.ready_parts(
                call.prompt_tokens, call.ordinal, slots, call.emitted, profile)
            region = call.ordinal
            support = {"phase": "READY", "mode": mode, "survivors": survivors,
                       "emitted": call.emitted}
            point_current = current * decode_seconds
        else:
            last = goal.history[-1]
            elapsed = goal.tool_elapsed / decode_seconds
            tail, survivors, interval, hi = self.away_base(
                last.prompt_tokens, last.ordinal, last.outputs, elapsed)
            tail = list(tail)
            if len(tail) % 2:
                tail.append(0.01)
            current = 0.0
            total_mean = math.fsum(tail)
            region = last.ordinal + 1
            mode = "survival_backoff"
            support = {"phase": "AWAY", "mode": mode, "elapsed_units": elapsed,
                       "survival_interval": interval, "max_tool_units": hi}
            point_current = 0.0
        q = dict(self.q_regions).get(region, self.q_cal) if conformal else 0.0
        current_padded = current + q * scale(current) if current else 0.0
        away_units = math.fsum(tail[::2])
        total = total_mean + q * scale(total_mean) + (
            self.away_q * scale(away_units) if conformal and away_units else 0.0)
        phases = []
        if goal.ready is not None:
            phases.append(["compute", current_padded])
            extra = max(0.0, total - current_padded - math.fsum(tail))
        else:
            extra = max(0.0, total - math.fsum(tail))
        future_compute = math.fsum(tail[1::2])
        for index in range(0, len(tail), 2):
            phases.append(["away", tail[index]])
            phases.append(["compute", tail[index + 1] + (
                extra * tail[index + 1] / future_compute if future_compute else 0.0)])
        phases = [(kind, max(0.0, value) * decode_seconds)
                  for kind, value in phases if value > 0]
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
            "support": support,
            "conformal": conformal,
            "q_region": region,
        }

    def to_dict(self):
        return {"records": self.records, "away_mean": self.away_mean,
                "away_q": self.away_q, "q_cal": self.q_cal,
                "q_regions": self.q_regions, "neighbors": self.neighbors,
                "version": self.version}

    @classmethod
    def from_dict(cls, value):
        records = tuple((dict(record) for record in value["records"]))
        for record in records:
            record["future"] = tuple(dict(item) for item in record["future"])
        return cls(records, value["away_mean"], value["away_q"], value["q_cal"],
                   tuple((int(key), item) for key, item in value["q_regions"]),
                   value["neighbors"], value["version"])
