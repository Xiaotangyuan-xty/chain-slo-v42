import hashlib
import json
from pathlib import Path
import socket
import subprocess

root = Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1')
stage = root / 'stage9_native_ac_gpu1_20261031_v1'
qualification = json.loads((stage / 'receipts/CPU_QUALIFICATION.json').read_text())
bad = []
for prefix, hashes in [(root, qualification['base_files']), (stage, qualification['stage_files'])]:
    for name, expected in hashes.items():
        p = prefix / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            bad.append(str(p))
protocol = json.loads((stage / 'PROTOCOL.json').read_text())
names = list(protocol['attempts'].values())
audits = {n: json.loads((stage / 'attempts' / n / 'common_audit_summary.json').read_text()) for n in names}
controllers = {n: json.loads((stage / 'attempts' / n / 'controller_receipt.json').read_text()) for n in names}
comparison = json.loads((stage / 'receipts/AC_COMPARISON.json').read_text())
baseline, other = comparison['arms']['A']['request_signatures'], comparison['arms']['C']['request_signatures']
token_matches = [k for k in baseline if baseline[k]['token_ids_sha256'] == other[k]['token_ids_sha256']]

attempt = stage / 'attempts' / protocol['attempts']['C']
reservations = [json.loads(line) for line in (attempt / 'reservation_events.jsonl').open()]
events = [json.loads(line) for line in (attempt / 'common_events.jsonl').open()]
index, mismatches, verified, swap_only, swap_in, swap_out = -1, [], 0, 0, 0, 0
for event in events:
    if event['kind'] != 'model_execute':
        continue
    begin = event['monotonic'] - event['elapsed_seconds']
    while index + 1 < len(reservations) and reservations[index+1]['monotonic'] <= begin:
        index += 1
    if index >= 0:
        unexpected = set(event['request_ids']) - set(reservations[index]['selected'])
        if unexpected:
            mismatches.append({'time': begin, 'request_ids': sorted(unexpected)})
        else:
            verified += 1
    swap_only += not event['request_ids']
    swap_in += event['swap_in_blocks']
    swap_out += event['swap_out_blocks']

gpu = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used','--format=csv,noheader'], text=True).strip()
processes = subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,used_memory','--format=csv,noheader'], text=True).strip()
ports = {}
for port in [18080, 65433]:
    try:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
        ports[str(port)] = True
    except OSError:
        ports[str(port)] = False
result = {'source_mismatches': bad, 'base_files': len(qualification['base_files']),
          'stage_files': len(qualification['stage_files']),
          'A_launcher_same_as_stage8': (stage/'programs/a_server.py').read_bytes() == (root/'stage8_native_abc_gpu1_20261031_v1/programs/a_server.py').read_bytes(),
          'gpu1': gpu, 'gpu1_processes': processes, 'ports_free': ports,
          'common_audits': audits, 'controllers': controllers,
          'same_ID_full_token_sequence_hash': {'matches': len(token_matches), 'total': len(baseline), 'match_keys': token_matches},
          'C_reservation_execution': {'events': len(reservations), 'verified_executor_calls': verified,
                                    'mismatches': mismatches, 'swap_only_calls': swap_only,
                                    'swap_in_blocks': swap_in, 'swap_out_blocks': swap_out},
          'scope': 'Six fixed DEV chains; fixed output-length replay. Full token equivalence is a separate check.'}
result['gate'] = 'PASS' if (not bad and not processes and gpu.endswith('0 MiB') and all(ports.values())
    and result['A_launcher_same_as_stage8'] and not mismatches and comparison['gate']=='PASS') else 'FAIL'
with (stage/'receipts/FINAL_AUDIT.json').open('x') as handle:
    json.dump(result,handle,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ['common_audits','controllers']},indent=2))
