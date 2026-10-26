#!/usr/bin/env python3
"""Verify exact released scientific, source, media and package-data bytes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'ARTIFACTS.json'


def files():
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        if not p.is_file() or p==MANIFEST:
            continue
        if any(x in {'.git','.venv','.pytest_cache','__pycache__','build','dist','snapshots'} or x.endswith('.egg-info') for x in relative.parts):
            continue
        if p.suffix in {'.pyc','.log'}:
            continue
        yield relative.as_posix(),p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='Create/update manifest after intentional release changes')
    args=parser.parse_args()
    if args.write:
        records={n:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for n,p in files()}
        MANIFEST.write_text(json.dumps({'version':'0.1.0','algorithm':'SHA256','files':records},indent=2)+'\n')
        print(f'Recorded {len(records)} artifact hashes.')
    data=json.loads(MANIFEST.read_text())
    errors=[]
    for name,record in data['files'].items():
        path=ROOT/name
        if not path.is_file(): errors.append(f'missing: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']: errors.append(f'changed: {name}')
    if errors: raise SystemExit('\n'.join(errors))
    print(f"Verified {len(data['files'])} exact artifact hashes.")


if __name__=='__main__':main()
