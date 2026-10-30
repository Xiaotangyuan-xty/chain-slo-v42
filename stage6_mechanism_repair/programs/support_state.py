from dataclasses import dataclass
import inspect


@dataclass
class SupportEntry:
    supported_through: float = -1.0
    unsupported_from: float | None = None

    def to_dict(self):
        return {"supported_through": self.supported_through,
                "unsupported_from": self.unsupported_from}


@dataclass(frozen=True)
class GoalSupport:
    key: str
    phase: str
    supported: bool
    source: str


@dataclass(frozen=True)
class SupportReport:
    denominator: int
    results: tuple

    @property
    def supported(self):
        return all(result.supported for result in self.results)

    @property
    def unsupported_keys(self):
        return [result.key for result in self.results if not result.supported]


class SupportState:
    def __init__(self, predictor):
        self.predictor = predictor
        self.entries = {}
        self.ready_calls = 0
        self.away_calls = 0
        self.ready_has_slots = "slots" in inspect.signature(predictor.ready_base).parameters

    def _check(self, key, position, query, phase):
        entry = self.entries.setdefault(key, SupportEntry())
        if entry.unsupported_from is not None and position >= entry.unsupported_from:
            return False, "cached_unsupported"
        try:
            query()
        except ValueError:
            entry.unsupported_from = position if entry.unsupported_from is None else min(
                entry.unsupported_from, position)
            return False, "predictor"
        entry.supported_through = max(entry.supported_through, position)
        return True, "predictor"

    def ready(self, prompt, ordinal, emitted, slots=1):
        self.ready_calls += 1
        key = ("READY", prompt, ordinal, slots) if self.ready_has_slots else ("READY", prompt, ordinal)
        if self.ready_has_slots:
            query = lambda: self.predictor.ready_base(prompt, ordinal, emitted, slots=slots)
        else:
            query = lambda: self.predictor.ready_base(prompt, ordinal, emitted)
        return self._check(key, emitted, query, "READY")

    def away(self, goal, decode_seconds, declared_prompt=None):
        last = goal.history[-1]
        prompt = last.prompt_tokens if declared_prompt is None else declared_prompt
        elapsed = goal.tool_elapsed / decode_seconds
        self.away_calls += 1
        return self._check(
            ("AWAY", prompt, last.ordinal, last.outputs), elapsed,
            lambda: self.predictor.away_base(prompt, last.ordinal,
                                             last.outputs, elapsed), "AWAY")

    def check_goals(self, goals, decode_seconds, compound_members=None, declared_inputs=None):
        compound_members = compound_members or {}
        declared_inputs = declared_inputs or {}
        results = []
        for goal in goals:
            if goal.state == "AWAY":
                key = f"away:{goal.program}"
                supported, source = self.away(goal, decode_seconds, declared_inputs.get(key))
            else:
                call = goal.ready
                key = call.key
                prompt = declared_inputs.get(key, call.prompt_tokens)
                slots = len(compound_members.get(key, (key,)))
                supported, source = self.ready(prompt, call.ordinal, call.emitted, slots)
            results.append(GoalSupport(key, goal.state, supported, source))
        return SupportReport(len(goals), tuple(results))

    @staticmethod
    def classify_audit(audit):
        fallback = audit.get("fallback")
        if audit.get("accepted") or fallback is None:
            return "success"
        if fallback == "no_model_gain":
            return "no_model_gain"
        text = str(fallback).lower()
        if "timeout" in text or "reference budget" in text:
            return "timeout"
        if "support" in text and any(word in text for word in
                                     ("unknown", "exhausted", "empty", "nonpositive")):
            return "unsupported"
        return "evaluation_error"

    def observe_audit(self, audit, goals, decode_seconds, compound_members=None, declared_inputs=None):
        disposition = self.classify_audit(audit)
        if disposition == "unsupported":
            self.check_goals(goals, decode_seconds, compound_members, declared_inputs)
        return disposition

    def fallback_action(self, calls, eligible, decode_seconds):
        candidates = [call for call in calls if call.key in eligible]
        if not candidates:
            raise ValueError("support fallback requires an eligible composite")
        call = min(candidates, key=lambda item: (item.level, -item.ready_wait, item.key))
        return (call.key, 4 * decode_seconds), {
            "pi0": call.key,
            "actions": [],
            "accepted": False,
            "fallback": "support_gate",
            "fallback_kind": "unsupported",
            "seconds": 0.0,
        }

    def snapshot(self):
        return {str(key): entry.to_dict() for key, entry in self.entries.items()}
