"""Explicit experiment-root binding; independent of per-card case depth."""
from pathlib import Path
import hashlib
import json

NATIVE_METRIC_SHA='208a817a166709ce69ea4eb9d85ba53f590d0b4494bc45fdd43b8d9d91928227'


def experiment_root(stage,root):
    if root is None:raise ValueError('Explicit experiment root is required')
    root=Path(root).resolve();protocol=json.loads((Path(stage)/'PROTOCOL.json').read_text())
    declared=protocol.get('experiment_root')
    if declared is not None and Path(declared).resolve()!=root:raise ValueError('Protocol experiment root mismatch')
    source=root/'repo/benchmark/benchmark_scheduler.py'
    if not source.is_file():raise ValueError('Explicit root lacks native metric source')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=NATIVE_METRIC_SHA:raise ValueError('Frozen native metric source changed')
    return root
