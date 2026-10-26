#!/usr/bin/env python3
"""Independent licensed-input replay/stability. No real separation truth assumed."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from etherglass.io import Capture,load_capture
from etherglass.analysis import analyze,public_report
from etherglass.classification import classify


def main():
    source=ROOT/'examples'/'real'/'radiopile-its-40.npz'
    samples=np.load(source,allow_pickle=False)['iq']
    report={'policy':'External inputs are test-only. No model or threshold changes based on these outcomes.',
            'radiopile':{'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'windows':len(samples),
                         'samples_per_window':500,'independent_sessions':'unknown; adjacent rows may be correlated',
                         'results':[],'scope':'Finite measured DSP and short-record abstention only. 500 samples <2048 classifier minimum; not a learned unknown-detection accuracy test.'}}
    for i,row in enumerate(samples):
        x=(row[0]+1j*row[1]).astype('complex64')
        a=public_report(analyze(Capture(x,25e6,5.9e9,name=f'RadioPile ITS window{i}')))
        json.dumps(a,allow_nan=False)
        report['radiopile']['results'].append({'index_in_excerpt':i,'finite_report':True,
            'classification':a['classification'],'peak_power_db':max(a['spectrum']['power_db'])})
    camras=ROOT/'examples'/'real'/'camras-danuri-65536.sigmf-data'
    if camras.exists():
        cap=load_capture(camras)
        # The symbol-rate below is an analysis hypothesis, NOT a source label.
        a=public_report(analyze(cap,target_center=0,symbol_rate=cap.sample_rate/8,method='hybrid'))
        json.dumps(a,allow_nan=False)
        rows=[]
        for start in range(0,len(cap.samples),2048):
            c=classify(cap.samples[start:start+2048],cap.sample_rate,cap.sample_rate/8)
            rows.append({'start_sample':start,**c})
        report['camras']={'sha256':cap.source_sha256,'samples':len(cap.samples),'recording_sessions':1,
            'analysis_prior':'Hypothesized8samples/symbol, centered0Hz; true modulation and symbol rate unknown.',
            'whole_capture_report_finite':True,'classification_windows':rows,
            'accepted_windows':sum(r['status']!='unknown' for r in rows),
            'separation_diagnostics':a['separation']['diagnostics'],
            'scope':'Actual-input numerical stability and conditional abstention; no clean reference, no separation quality or classification accuracy asserted.'}
    out=ROOT/'research'/'external_results.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x not in ['results','classification_windows']} if isinstance(v,dict) else v for k,v in report.items()},indent=2))


if __name__=='__main__': main()
