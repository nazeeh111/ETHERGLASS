#!/usr/bin/env python3
"""Copy release inference artifacts into wheel package data; verify exact bytes."""
import hashlib
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
dest=ROOT/'src'/'etherglass'/'resources'/'models'
dest.mkdir(parents=True,exist_ok=True)
for path in [*(ROOT/'models').glob('*.pt'),*(ROOT/'models').glob('*.json'),*(ROOT/'models').glob('seed*/hybrid.pt')]:
    # Evaluation features/CSVs and all ablation seed models stay in repository,
    # while the wheel contains inference models and three hybrid uncertainty seeds.
    out=dest/path.relative_to(ROOT/'models')
    out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,out)
    assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(out.read_bytes()).digest()
print('Copied exact release model bytes into package resources.')
