#!/usr/bin/env python3
"""Render editable SVGs and PNGs from actual inputs, checkpoints and measured CSVs."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from etherglass.analysis import demo_capture
from etherglass.separation import separate
from etherglass.metrics import waveform_metrics

PAPER='#eeede6'; INK='#252c2b'; COPPER='#a4472a'; INDIGO='#474b97'; TEAL='#198a8e'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.facecolor':PAPER,
    'figure.facecolor':PAPER,'axes.edgecolor':'#b9b9af','axes.labelcolor':INK,
    'text.color':INK,'xtick.color':INK,'ytick.color':INK,'axes.spines.top':False,
    'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':PAPER})


def save(fig,name):
    for fmt in ['svg','png']:
        fig.savefig(ROOT/'assets'/f'{name}.{fmt}',dpi=170,bbox_inches='tight',metadata={'Creator':'ETHERGLASS actual experiment plotting'})
    plt.close(fig)


def main():
    cap,truth=demo_capture()
    methods=['matched','wiener','notch','hybrid']
    outputs={m:separate(cap.samples,cap.sample_rate,symbol_rate=125000,method=m)['components'][0] for m in methods}
    records={m:waveform_metrics(truth,x) for m,x in {'mixture':cap.samples,**outputs}.items()}
    (ROOT/'assets'/'demo-metrics.json').write_text(json.dumps({'seed':870001,'n':len(truth),'metrics':records},indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,6.7),gridspec_kw={'height_ratios':[1,1.2]})
    t=np.arange(600)/cap.sample_rate*1000
    axes[0,0].plot(t,cap.samples[:600].real,color=COPPER,lw=.8,label='Mixture I')
    axes[0,0].plot(t,truth[:600].real,color=TEAL,lw=.8,alpha=.9,label='Clean target I')
    axes[0,0].set(xlabel='Time (ms)',ylabel='Input amplitude',title='The observed mixture')
    axes[0,0].legend(frameon=False,ncols=2,fontsize=8)
    axes[0,1].plot(t,truth[:600].real,color=TEAL,lw=1,label='Clean reference I')
    axes[0,1].plot(t,outputs['hybrid'][:600].real,color=INDIGO,lw=.85,alpha=.8,label='Recovered target I')
    axes[0,1].set(xlabel='Time (ms)',ylabel='Input amplitude',title='The conditional estimate')
    axes[0,1].legend(frameon=False,ncols=2,fontsize=8)
    for name,x,c in [('Mixture',cap.samples,COPPER),('Clean target',truth,TEAL),('Hybrid estimate',outputs['hybrid'],INDIGO)]:
        f,p=signal.welch(x.astype('complex128'),fs=cap.sample_rate,nperseg=1024,return_onesided=False,detrend=False)
        axes[1,0].plot(np.fft.fftshift(f)/1000,10*np.log10(np.maximum(np.fft.fftshift(p),1e-20)),color=c,lw=1,label=name)
    axes[1,0].set(xlabel='Baseband frequency (kHz)',ylabel='PSD (dB input unit²/Hz)',xlim=(-150,150),title='Same received target, less interference')
    axes[1,0].legend(frameon=False,fontsize=8)
    names=list(records);values=[records[n]['si_sdr_db'] for n in names]
    bars=axes[1,1].barh(names,values,color=[COPPER,'#adb0a3','#b6b8ae',TEAL,INDIGO],height=.55)
    axes[1,1].axvline(0,color='#a9aba1',lw=.7)
    axes[1,1].set(xlabel='Complex waveform SI-SDR (dB)',title='Fixed reference capture / 870001')
    for bar,v in zip(bars,values): axes[1,1].text(v+.2,bar.get_y()+bar.get_height()/2,f'{v:.2f}',va='center',fontsize=8)
    fig.suptitle('ETHERGLASS / A reproducible separation observation',x=.055,ha='left',fontsize=18,fontweight='medium')
    fig.text(.055,.015,'QPSK + chirp · seed870001 · SNR20dB · SIR−6dB · complex scalar alignment; no delay search · all traces calculated from actual samples',fontsize=8,color='#62685f')
    fig.tight_layout(rect=(0,.035,1,.93));save(fig,'signal-comparison')
    cover=plt.figure(figsize=(10,5.25),facecolor=PAPER)
    cover.text(.06,.87,'E T H E R G L A S S',fontsize=30,fontweight='bold',color=INK)
    cover.text(.06,.75,'The structure beneath the interference.',fontsize=20,color=INK)
    cover.text(.06,.67,'RF SIGNAL OBSERVATORY / KNOWN PRIORS / MEASURED EVIDENCE',fontsize=9,color=COPPER)
    axis=cover.add_axes([.06,.18,.88,.38])
    for x,c,label in [(cap.samples,COPPER,'Observed mixture'),(outputs['hybrid'],INDIGO,'Recovered target')]:
        f,p=signal.welch(x.astype('complex128'),fs=cap.sample_rate,nperseg=1024,return_onesided=False,detrend=False)
        axis.plot(np.fft.fftshift(f)/1000,10*np.log10(np.maximum(np.fft.fftshift(p),1e-20)),color=c,lw=1.2,label=label)
    axis.set(xlim=(-130,130),xlabel='Baseband frequency (kHz)',ylabel='PSD (dB unit²/Hz)')
    axis.legend(frameon=False,ncols=2,fontsize=8)
    cover.text(.06,.045,'Fixed synthetic reference · actual input and trained output · CPU replay · SigMF exports',fontsize=8,color=INK)
    cover.savefig(ROOT/'assets'/'social-preview.png',dpi=120)
    cover.savefig(ROOT/'assets'/'social-preview.svg')
    plt.close(cover)
    result_path=ROOT/'research'/'results.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        methods=['mixture','matched','wiener','notch','unet','pretrained','scratch','no_dsp','hybrid']
        fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
        for ax,cell,title in zip(axes,['nominal','shift'],['Stationary-tone holdout','Chirp + channel-shift holdout']):
            table=result['methods'][cell]
            present=[m for m in methods if m in table]
            vals=[table[m]['si_sdr_db'] for m in present]
            ax.barh(present,vals,color=[INDIGO if m=='hybrid' else TEAL if m=='notch' else '#b7b9af' for m in present],height=.65)
            for i,m in enumerate(present):
                means=table[m].get('training_seed_mean_si_sdr_db',{})
                if len(means)>1: ax.scatter(list(means.values()),np.full(len(means),i),s=13,c=INK,zorder=3)
            ax.axvline(0,c='#a9aba1',lw=.7)
            ax.set(xlabel='Mean complex waveform SI-SDR (dB)',title=title)
        fig.suptitle('ETHERGLASS / Matched priors, independent examples',x=.05,ha='left',fontsize=17)
        fig.text(.05,.015,f"{result['examples_per_cell']} generated examples per cell · dots show training-seed means · source: research/evaluation.csv · no real-recording quality claim",fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.91));save(fig,'benchmark')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
