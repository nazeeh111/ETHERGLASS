#!/usr/bin/env python3
"""Rebuild the fixed CC0 synthetic observation with exact clean components."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from etherglass.synthetic import generate_example
from etherglass.io import encode_sigmf

out=ROOT/'examples'/'reference'
out.mkdir(parents=True,exist_ok=True)
ex=generate_example(870001,n=16384,split='test',snr_db=20,sir_db=-6,family='QPSK')
manifest={'seed':870001,'selected_by_performance':False,'generator_metadata':ex['metadata'],'files':{}}
for name in ['mixture','target','interference','noise']:
    raw,meta=encode_sigmf(ex[name],1e6,0,description=f'ETHERGLASS fixed synthetic reference: {name}',
        license='https://creativecommons.org/publicdomain/zero/1.0/',
        provenance={'generator':ex['metadata'],'component':name,'ground_truth':name!='mixture'})
    for suffix,data in [('sigmf-data',raw),('sigmf-meta',(json.dumps(meta,indent=2)+'\n').encode())]:
        p=out/f'{name}.{suffix}';p.write_bytes(data)
        manifest['files'][p.name]=hashlib.sha256(data).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'README.md').write_text('# Fixed reference observation\n\nQPSK+chirp seed870001,16,384complex samples,1Msps,125kbaud,SNR20dB,SIR−6dB. This seed was fixed before evaluating its performance.\n\nOriginal synthetic data: CC0-1.0. `target` is the clean received waveform; `interference` and `noise` are exact generated components. They sum to `mixture` within float32 precision. Metadata preserves generator/channel parameters and symbols are random, not private messages. Clean references are used only for supervised training/evaluation, never provided to the inference model.\n\nRebuild with `python scripts/generate_reference.py`. Manifest contains actual SHA256 hashes.\n')
print('Wrote four actual components and their provenance.')
