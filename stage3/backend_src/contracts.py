"""Immutable online information and plan/commit cache contracts. No GPU imports."""
from dataclasses import dataclass, replace
from collections import Counter
import bisect, hashlib, json, math, random

BETAS = (.25, .5, 1., 2., 8.)
CAL_SEEDS = (615101, 615102)
TEST_SEEDS = tuple(range(615201, 615213))

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def quantile(xs, p=.9):
    if not xs: raise ValueError('empty empirical distribution')
    return sorted(xs)[max(0, math.ceil(p * len(xs)) - 1)]

@dataclass(frozen=True)
class CacheEntry:
    key: str
    gpu: int = 0
    host: int = 0
    gpu_touch: int = 0
    host_touch: int = 0
    pending: int = 0

@dataclass(frozen=True)
class CacheView:
    gpu_capacity: int
    host_capacity: int
    entries: tuple

    def get(self, key):
        return next((e for e in self.entries if e.key == key), CacheEntry(key))

    @property
    def free(self):
        return self.gpu_capacity - sum(e.gpu+e.pending for e in self.entries)

class CacheLedger:
    """Persistent-token quota; planning never publishes uncomputed KV hits.

    Host and GPU LRU are common to every scheduling arm. Actual tensor producers
    must finish before commit(). Temporary forward buffers are outside this
    declared KV quota and are reported separately, not called physical OOM.
    """
    def __init__(self, gpu_capacity, host_capacity):
        if gpu_capacity <= 0 or host_capacity < 0: raise ValueError('capacity')
        self.gpu_capacity, self.host_capacity = gpu_capacity, host_capacity
        self.entries, self.clock, self.events = {}, 0, []

    def view(self):
        return CacheView(self.gpu_capacity, self.host_capacity,
                         tuple(sorted(self.entries.values(), key=lambda e: e.key)))

    def plan(self, key, required):
        if required < 0 or required > self.gpu_capacity:
            raise ValueError('request exceeds registered KV quota')
        entries = dict(self.entries)
        old = entries.get(key, CacheEntry(key))
        need = max(0, required - old.gpu)
        victims, actions = [], []
        used = sum(e.gpu+e.pending for e in entries.values())
        for initial in sorted(entries.values(), key=lambda e: (e.gpu_touch, e.key)):
            e = entries[initial.key]
            if used + need <= self.gpu_capacity: break
            if e.key == key or not e.gpu or e.pending: continue
            victims.append(e.key)
            used -= e.gpu
            if e.gpu <= self.host_capacity:
                host_used = sum(v.host for v in entries.values()) - e.host
                for h in sorted(entries.values(), key=lambda v: (v.host_touch, v.key)):
                    if host_used + e.gpu <= self.host_capacity: break
                    if h.key in (e.key, key) or not h.host: continue
                    actions.append(('drop_host', h.key, h.host))
                    host_used -= h.host
                    entries[h.key] = replace(h, host=0)
                if host_used + e.gpu <= self.host_capacity:
                    actions.append(('swap_out', e.key, e.gpu))
                    entries[e.key] = replace(e, gpu=0, host=e.gpu)
                    continue
            actions.append(('drop_gpu', e.key, e.gpu))
            entries[e.key] = replace(e, gpu=0)
        if used + need > self.gpu_capacity: raise RuntimeError('no legal admission')
        return tuple(actions)

    def commit(self, event, key, tokens=0):
        self.clock += 1
        e = self.entries.get(key, CacheEntry(key))
        if event == 'swap_out':
            if tokens != e.gpu: raise RuntimeError('swap producer length mismatch')
            e = replace(e, gpu=0, host=tokens, host_touch=self.clock)
        elif event == 'drop_gpu': e = replace(e, gpu=0)
        elif event == 'drop_host': e = replace(e, host=0)
        elif event == 'gpu_produced':
            if tokens < e.gpu: raise RuntimeError('prefix watermark regressed')
            e = replace(e, gpu=tokens, gpu_touch=self.clock, pending=0)
        elif event == 'reserve_copy':
            if e.gpu or e.pending or not 0 < tokens <= e.host:
                raise RuntimeError('invalid prefetch reservation')
            e = replace(e, pending=tokens)
        elif event == 'touch': e = replace(e, gpu_touch=self.clock)
        elif event == 'release':
            if e.pending: raise RuntimeError('release before producer completion')
            e = replace(e, gpu=0, host=0)
        else: raise ValueError(event)
        self.entries[key] = e
        if sum(v.gpu+v.pending for v in self.entries.values()) > self.gpu_capacity:
            raise RuntimeError('GPU quota exceeded')
        if sum(v.host for v in self.entries.values()) > self.host_capacity:
            raise RuntimeError('host quota exceeded')
        self.events.append((event, key, tokens, self.clock))

@dataclass(frozen=True)
class CallView:
    key: str
    program: str
    ordinal: int
    prompt_tokens: int
    emitted: int
    produced_prefix: int
    level: int
    quantum_used: float
    ready_wait: float
    service: float
    virtual_service: float
    running: bool
    historical_wait: float = 0.

@dataclass(frozen=True)
class OnlineView:
    now: float
    calls: tuple
    cache: CacheView
    program_states: tuple

@dataclass(frozen=True)
class Intent:
    key: str
    horizon: float
    source: str

@dataclass(frozen=True)
class CostProfile:
    """Deployment-fit seconds; never trace token ratios as time coefficients."""
    prefill_points: tuple
    decode_seconds: float
    copy_seconds_per_token: float
    relative_error: float
    coefficients: tuple = ()

    def prefill(self, new, cached=0):
        if new <= 0: return 0.
        pts = self.prefill_points
        length = new + cached
        if self.coefficients:
            if length > max(p[0]+p[1] for p in pts):
                raise ValueError('prefill outside profile support')
            features=(1.,new,new*cached,new*new,cached,cached*cached)
            value=sum(a*b for a,b in zip(self.coefficients,features))
            if value <= 0 or not math.isfinite(value): raise ValueError('invalid fitted cost')
            return value
        if cached: raise ValueError('cache-hit profile required')
        if length > pts[-1][0]: raise ValueError('prefill outside profile support')
        x = [p[0] for p in pts]
        idx = min(len(pts)-1, max(1, bisect.bisect_left(x, length)))
        a, b = pts[idx-1], pts[idx]
        total = a[1] + (b[1]-a[1]) * (length-a[0])/(b[0]-a[0])
        return max(0., total)

    def recovery(self, entry, needed):
        if entry.gpu >= needed: return 0.
        available = max(entry.gpu, min(entry.host, needed))
        return ((available-entry.gpu) * self.copy_seconds_per_token
                + self.prefill(needed-available, available))

@dataclass(frozen=True)
class FrozenPredictor:
    version: str
    groups: tuple
    provenance: tuple

    @staticmethod
    def bucket(prompt): return min(3, int(math.log2(max(128, prompt) / 128)))

    @classmethod
    def fit(cls, rows, allowed_seeds=CAL_SEEDS):
        if not rows or any(r['seed'] not in allowed_seeds for r in rows):
            raise ValueError('predictor training requires registered DEV only')
        groups = {}
        for r in rows:
            k = (cls.bucket(r['prompt_tokens']), min(3, r['ordinal']))
            groups.setdefault(k, []).append((r['outputs'], bool(r['continues']),
                                              r['next_prompt'], r['away']))
        return cls('empirical-survival-v1', tuple((k, tuple(v)) for k,v in sorted(groups.items())),
                   tuple(sorted(set(r['seed'] for r in rows))))

    def records(self, prompt, ordinal):
        k = (self.bucket(prompt), min(3, ordinal))
        groups = dict(self.groups)
        if k not in groups: raise ValueError('predictor support unknown')
        return groups[k]

    def remaining(self, prompt, ordinal, emitted, rng):
        rows = [r for r in self.records(prompt, ordinal) if r[0] > emitted]
        if not rows: raise ValueError('conditional survival support exhausted')
        r = rows[rng.randrange(len(rows))]
        return r[0]-emitted, r

    def p90(self, prompt, ordinal, emitted):
        vals = [r[0]-emitted for r in self.records(prompt, ordinal) if r[0] > emitted]
        return quantile(vals)

    def to_dict(self):
        return {'version':self.version, 'groups':self.groups, 'provenance':self.provenance,
                'P90_is_empirical_quantile_not_a_guaranteed_upper_bound':True}

    @classmethod
    def from_dict(cls, d):
        return cls(d['version'], tuple((tuple(k),tuple(tuple(r) for r in rows)) for k,rows in d['groups']),
                   tuple(d['provenance']))
