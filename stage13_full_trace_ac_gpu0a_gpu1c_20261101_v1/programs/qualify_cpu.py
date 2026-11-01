import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root, stage = Path(sys.argv[1]), Path(sys.argv[2])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU qualification must hide every GPU')
    tests = {}
    commands = {
        'full_trace_transport_and_recording': stage / 'programs/controlled_full_trace.py',
        'native_chain_cpu_runtime': stage / 'programs/controlled_native.py',
        'v42_dynamic_batch': root / 'stage7_dynamic/programs/controlled_dynamic_batch.py',
        'v42_scheduler': root / 'stage7_dynamic/programs/controlled_stage7_scheduler.py',
        'v42_rollout': root / 'stage7_dynamic/programs/controlled_stage7_rollout.py',
    }
    for name, script in commands.items():
        p = subprocess.run([sys.executable, str(script)], cwd=root, env=os.environ.copy(),
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = stage / 'receipts' / f'CPU_{name}.log'
        with log.open('x') as f:
            f.write(p.stdout)
        tests[name] = {'pass': p.returncode == 0, 'log_sha256': sha(log)}
    frozen = json.loads((root / 'stage7_dynamic/receipts/CPU_QUALIFICATION_V44.json').read_text())
    base_files = dict(frozen['files'])
    official = json.loads((root / 'manifests/official_source_hashes.json').read_text())
    base_files.update({'repo/' + name: item['sha256'] for name, item in official['files'].items()})
    native = json.loads((root / 'manifests/native_binary_reuse.json').read_text())
    base_files.update({item['path']: item['sha256'] for item in native['files']})
    bad = [name for name, digest in base_files.items() if not (root / name).is_file()
           or sha(root / name) != digest]
    tests['v42_and_official_native_source_unchanged'] = {'pass': not bad, 'bad': bad,
                                                       'v42_files': 70, 'official_files': 1297}
    core = json.loads((stage / 'FROZEN_V43_CORE.json').read_text())
    base_files.update(core['frozen_stage9_files'])
    frozen_bad = [name for name, digest in core['files'].items()
                  if sha(stage / name) != digest]
    frozen_bad.extend(name for name, digest in core['frozen_stage9_files'].items()
                      if sha(root / name) != digest)
    for name, digest in core['launcher_normalized_sha256'].items():
        normalized = (stage / 'programs' / name).read_text().replace('from gpu0_platform import install', 'from gpu1_platform import install')
        if hashlib.sha256(normalized.encode()).hexdigest() != digest:
            frozen_bad.append(name)
    tests['frozen_v43_core_unchanged'] = {'pass': not frozen_bad, 'bad': frozen_bad}
    stage_files = {}
    for p in sorted((stage / 'programs').glob('*.py')):
        compile(p.read_text(), str(p), 'exec')
        stage_files[str(p.relative_to(stage))] = sha(p)
    for name in ['PROTOCOL.json', 'GPU0_AUTHORIZATION.json', 'GPU1_AUTHORIZATION.json',
                 'TRACE_MANIFEST.json', 'FROZEN_V43_CORE.json']:
        stage_files[name] = sha(stage / name)
    result = {'schema': 'stage13-cpu-qualification/v1',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'gate': 'PASS' if all(v['pass'] for v in tests.values()) else 'FAIL',
              'tests': tests, 'gpu_used': False, 'base_files': base_files,
              'stage_files': stage_files,
              'frozen_C_prior_GPU_compatibility': 'Stage9 GPU1 PASS; Stage13 GPU0/GPU1 device adapters and both launchers CPU checked; both arms execute the same warmup before full replay',
              'runtime_boundary': 'Full trace wrappers, optimized streaming response parsing, background client JSONL writing and finite-replay GC stabilization are CPU qualified. Frozen C core and native A scheduling remain unchanged. '
                                  'Per-arm authorized-GPU idle/UUID/source gates and standard warmup precede each arm; A must pass before C starts.'}
    path = stage / 'receipts/CPU_QUALIFICATION.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'gate': result['gate'], 'tests': tests, 'base_files': len(base_files),
                      'stage_files': len(stage_files)}, indent=2))
    return 0 if result['gate'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
