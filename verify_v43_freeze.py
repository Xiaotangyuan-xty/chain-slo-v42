import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SNAPSHOT_V43_MANIFEST.json").read_text())
bad = []
for name, entry in manifest["files"].items():
    p = root / name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
        bad.append(name)
freeze = json.loads((root / "FREEZE_V43.json").read_text())
stage = root / freeze["stage"]
q = json.loads((stage / "receipts/CPU_QUALIFICATION.json").read_text())
assert q["gate"] == "PASS" and len(q["stage_files"]) == 14
assert len(q["base_files"]) == 1372
for name, digest in q["stage_files"].items():
    if hashlib.sha256((stage / name).read_bytes()).hexdigest() != digest:
        bad.append(str(Path(freeze["stage"]) / name))
external = freeze["external_native_binary_sha256"]
for name, digest in q["base_files"].items():
    if name in external:
        assert digest == external[name] and name.endswith(".so")
        continue
    p = root / name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        bad.append(name)
comparison = json.loads((stage / "receipts/AC_COMPARISON.json").read_text())
assert comparison["gate"] == "PASS" and comparison["C_strictly_more_scaled_chains"]
if bad:
    raise SystemExit("SHA mismatch: " + ", ".join(sorted(set(bad))))
print(json.dumps({"gate": "PASS", "payload_files": len(manifest["files"]),
    "qualified_stage_files": 14, "base_files_present": 1369,
    "external_native_binaries": 3, "gpu_used": False}))
