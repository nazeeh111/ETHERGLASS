#!/usr/bin/env python3
"""Verify exact released scientific, source, media and package-data bytes."""
import argparse
import hashlib
import json
import tomllib
from pathlib import Path, PurePosixPath
from email.parser import BytesParser

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'ARTIFACTS.json'


def generated_sdist_root_metadata():
    """Check source inventory plus generated Name/version/README fields.

    Other generated headers are not authenticated here; the complete archive
    checksum identifies their shipped bytes.
    """
    info=ROOT/'PKG-INFO';config=ROOT/'setup.cfg';egg=ROOT/'src/etherglass.egg-info'
    paths=(info,config,egg/'PKG-INFO',egg/'SOURCES.txt',ROOT/'pyproject.toml',ROOT/'README.md')
    if not all(p.is_file() and p.resolve().is_relative_to(ROOT) for p in paths):
        return set()
    try:
        if config.read_bytes()!=b'[egg_info]\ntag_build = \ntag_date = 0\n\n':
            return set()
        payload=info.read_bytes()
        if payload!=(egg/'PKG-INFO').read_bytes():
            return set()
        project=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
        headers=BytesParser().parsebytes(payload,headersonly=True)
        if (headers.defects or len(headers.get_all('Name',[]))!=1
                or len(headers.get_all('Version',[]))!=1
                or headers['Name']!=project['name'] or headers['Version']!=project['version']
                or payload.partition(b'\n\n')[2]!=(ROOT/'README.md').read_bytes()):
            return set()
        entries=set((egg/'SOURCES.txt').read_text().splitlines())
        required={'ARTIFACTS.json','README.md','pyproject.toml','src/etherglass/__init__.py',
                  'src/etherglass.egg-info/PKG-INFO','src/etherglass.egg-info/SOURCES.txt'}
        required.update(json.loads(MANIFEST.read_text())['files'])
        if not required<=entries:
            return set()
        for name in entries:
            relative=PurePosixPath(name);path=ROOT/name
            if (not name or relative.is_absolute() or '..' in relative.parts
                    or not path.resolve().is_relative_to(ROOT) or not path.is_file()):
                return set()
    except (OSError,ValueError,KeyError,TypeError):
        return set()
    return {'PKG-INFO','setup.cfg'}


def files():
    generated=generated_sdist_root_metadata()
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        if not p.is_file() or p==MANIFEST or relative.as_posix() in generated:
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
        MANIFEST.write_text(json.dumps({'version':tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version'],'algorithm':'SHA256','files':records},indent=2)+'\n')
        print(f'Recorded {len(records)} artifact hashes.')
    data=json.loads(MANIFEST.read_text())
    errors=[]
    eligible={name for name,path in files()}
    recorded=set(data['files'])
    errors.extend(f'unlisted: {name}' for name in sorted(eligible-recorded))
    errors.extend(f'missing or excluded: {name}' for name in sorted(recorded-eligible))
    for name,record in data['files'].items():
        path=ROOT/name
        if not path.is_file(): errors.append(f'missing: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']: errors.append(f'changed: {name}')
    if errors: raise SystemExit('\n'.join(errors))
    print(f"Verified {len(data['files'])} exact artifact hashes.")


if __name__=='__main__':main()
