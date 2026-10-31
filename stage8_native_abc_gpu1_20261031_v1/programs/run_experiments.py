import json
from pathlib import Path
import subprocess
import sys


def main():
    root, stage = Path(sys.argv[1]), Path(sys.argv[2])
    protocol = json.loads((stage / 'PROTOCOL.json').read_text())
    python = json.loads((root / 'PREPARE.json').read_text())['python']
    names = protocol['attempts']
    for mode, arm in [('compat', 'A'), ('full', 'A'), ('full', 'B'), ('full', 'C')]:
        name = names['compat'] if mode == 'compat' else names[arm]
        print(json.dumps({'phase': 'START', 'arm': arm, 'mode': mode, 'attempt': name}), flush=True)
        subprocess.run([python, str(stage / 'programs/controller.py'), str(root), str(stage),
                        arm, name, mode], check=True)
        subprocess.run([python, str(stage / 'programs/offline_compare.py'), str(root), str(stage),
                        'arm', name, mode], check=True)
        print(json.dumps({'phase': 'PASS', 'arm': arm, 'mode': mode, 'attempt': name}), flush=True)
    subprocess.run([python, str(stage / 'programs/offline_compare.py'), str(root), str(stage),
                    'abc', names['A'], names['B'], names['C']], check=True)


if __name__ == '__main__':
    main()
