#!/usr/bin/env python3
"""Calibrate a 90% ensemble-scaled raw waveform-RMS error bound on untouched seeds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from etherglass.synthetic import generate_example
from etherglass.separation import separate
from etherglass.uncertainty import ensemble_indicator,source_fingerprints


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,default=96)
    args=p.parse_args()
    if not 9 <= args.count <= 10000:
        p.error('--count must be9..10000: finite90% conformal rank and disjoint calibration/test seeds are required.')
    rows=[]
    for group,start in [('calibration',900000),('test',910000)]:
        for i in range(args.count):
            ex=generate_example(start+i,n=4096,split='test')
            y=separate(ex['mixture'],1e6,symbol_rate=125000)['components'][0]
            u=ensemble_indicator(ex['mixture'],1e6,0,125000,estimate=y)
            if u['status']=='unavailable':
                raise RuntimeError(u['reason'])
            error=float(np.mean(np.abs(y-ex['target'])**2))
            rows.append({'split':group,'seed':start+i,'raw_error_power':error,
                         'ensemble_scale_power':u['error_scale_power'],
                         'nonconformity':error/u['error_scale_power'],
                         'relative_disagreement_rms':u['relative_disagreement_rms']})
    calibration=[r for r in rows if r['split']=='calibration']
    # Finite-sample split-conformal rank, without fitting against the test set.
    rank=int(np.ceil((len(calibration)+1)*.90))
    q=float(sorted(r['nonconformity'] for r in calibration)[rank-1])
    for r in rows:
        r['upper_error_power']=q*r['ensemble_scale_power']
        r['covered']=r['raw_error_power']<=r['upper_error_power']
    test=[r for r in rows if r['split']=='test']
    model={'version':'0.1.0','nominal_coverage':.90,'quantile':q,'calibration_examples':len(calibration),
           'rank':rank,'calibration_seeds':[900000,900000+args.count-1],
           'scope':'Synthetic shifted PSK+chirp distribution: SNR10..30dB, SIR-12..8dB, RRC rolloff.35, 8 samples/symbol, recorded CFO/timing/multipath distributions in synthetic.py. Raw received target waveform error, no alignment.',
           'scale':'mean per-sample variance across three hybrid training seeds +0.01*mixture power',
           'inference_source_sha256':source_fingerprints(),
           'checkpoint_sha256':{str(s):hashlib.sha256((ROOT/'models'/f'seed{s}'/'hybrid.pt').read_bytes()).hexdigest() for s in [0,1,2]}}
    (ROOT/'models'/'uncertainty.json').write_text(json.dumps(model,indent=2)+'\n')
    report={'model':model,'test_seeds':[910000,910000+args.count-1],
            'test_examples':len(test),'observed_coverage':sum(r['covered'] for r in test)/len(test),
            'mean_rms_upper':float(np.mean([np.sqrt(r['upper_error_power']) for r in test])),
            'mean_actual_rms_error':float(np.mean([np.sqrt(r['raw_error_power']) for r in test])),
            'limitations':['Only a finite synthetic marginal-coverage test; no real-RF coverage guarantee.',
                           'Same-model errors may be correlated. A numerical interval is not evidence of identifiability.']}
    (ROOT/'research'/'uncertainty_results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (ROOT/'research'/'uncertainty_evaluation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
