import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SNAPSHOT_MANIFEST.json").read_text())
bad = []
for name, item in manifest["files"].items():
    p = root / name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
        bad.append(name)
q = json.loads((root / "stage7_dynamic/receipts/CPU_QUALIFICATION_V44.json").read_text())
assert q["gate"] == "PASS" and q["mechanism_version"] == "v3_chain_dynamic_v42"
assert len(q["files"]) == 70
for name, digest in q["files"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        bad.append(name)
official = json.loads((root / "manifests/official_source_hashes.json").read_text())
assert len(official["files"]) == 1297
for name, item in official["files"].items():
    if hashlib.sha256((root / "repo" / name).read_bytes()).hexdigest() != item["sha256"]:
        bad.append("repo/" + name)
if bad:
    raise SystemExit("SHA mismatch: " + ", ".join(sorted(set(bad))))
print(json.dumps({"gate": "PASS", "snapshot_files": len(manifest["files"]),
                  "qualified_files": 70, "official_source_files": 1297,
                  "mechanism_version": "v3_chain_dynamic_v42", "gpu_used": False}))
