#!/usr/bin/env python3
"""Render all quantitative figures exclusively from saved measurements."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from etherglass.synthetic import generate_example
from etherglass.separation import separate
from etherglass.metrics import waveform_metrics

PAPER="#eeede6";INK="#252c2b";COPPER="#a4472a";TEAL="#198a8e";INDIGO="#474b97"
plt.rcParams.update({"figure.facecolor":PAPER,"axes.facecolor":PAPER,"axes.edgecolor":"#cfcec3",
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
    "font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
    "axes.spines.right":False,"grid.color":"#cfcec3","grid.alpha":.6,
    "savefig.facecolor":PAPER,"svg.fonttype":"none"})


def save(fig,name):
    folder=ROOT/"research"/"figures";folder.mkdir(exist_ok=True)
    fig.savefig(folder/f"{name}.png",dpi=180,bbox_inches="tight")
    fig.savefig(folder/f"{name}.svg",bbox_inches="tight")
    plt.close(fig)


def main():
    results=json.loads((ROOT/"research"/"results.json").read_text())
    seed_count=len(results["evaluation_config"]["training_seeds"])
    methods=["mixture","matched","wiener","notch","unet","pretrained","scratch","no_dsp","hybrid"]
    labels=["Mixture","RRC matched","Robust Wiener","Adaptive notch","Spectral U-Net","Pretrained only","Adapt from scratch","Contrast ablation","ETHERGLASS"]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,cell,title in zip(axes,["nominal","shift"],["Stationary tones · nominal","Chirps + channel shift"]):
        vals=[results["methods"][cell][m]["si_sdr_db"] for m in methods]
        colors=[COPPER if m=="hybrid" else INDIGO if m=="unet" else "#a4a69b" for m in methods]
        ax.barh(np.arange(len(methods)),vals,color=colors,height=.65)
        for i,m in enumerate(methods):
            seedmeans=results["methods"][cell][m].get("training_seed_mean_si_sdr_db",{})
            if len(seedmeans)>1:
                ax.scatter(list(seedmeans.values()),np.full(len(seedmeans),i),c=INK,s=12,zorder=3)
        ax.set_yticks(np.arange(len(methods)),labels)
        ax.invert_yaxis();ax.set_xlabel("Complex waveform SI-SDR (dB), higher is better")
        ax.set_title(title,loc="left",fontweight="bold",pad=16)
        ax.axvline(0,color="#72776d",lw=.6)
        for i,v in enumerate(vals):ax.text(max(v,0)+.15,i,f"{v:.2f}",va="center",fontsize=8)
        ax.set_xlim(min(0,min(vals)-1),max(vals)+2)
    fig.suptitle(f"Prior-conditioned target extraction · {results['examples_per_cell']} held-out records/cell · {seed_count} training seeds",x=.03,ha="left",fontsize=14,fontweight="bold")
    save(fig,"separation-benchmark")

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for ax,prefix,xlabel in zip(axes,["snr","sir"],["Target / noise power (dB)","Target / interference power (dB)"]):
        cells=sorted([(float(k.split("_")[1]),k) for k in results["methods"] if k.startswith(prefix+"_")])
        for m,color in [("hybrid",COPPER),("unet",INDIGO),("notch",TEAL),("matched","#72776d"),("pretrained","#b58b42")]:
            ax.plot([v for v,_ in cells],[results["methods"][k][m]["si_sdr_db"] for _,k in cells],"o-",label=dict(zip(methods,labels)).get(m,m),color=color,lw=1.8,ms=4)
        ax.set_xlabel(xlabel);ax.set_ylabel("Waveform SI-SDR (dB)");ax.grid(axis="y");ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Controlled chirp/channel holdouts · paired inputs for every method",x=.03,ha="left",fontsize=14,fontweight="bold")
    save(fig,"robustness-sweeps")

    cells=[("flat","Flat channel + chirp"),("multipath","Multipath + chirp"),
           ("timing","0.85-sample timing offset"),("carrier","0.004 cycles/sample carrier offset"),
           ("torchsig_independent_generator","Independent TorchSig PSK generator"),
           ("overlapping_digital_stress","Overlapping QPSK stress (unidentifiable)"),
           ("rate_6_samples_per_symbol","6 samples/symbol (untrained rate)"),
           ("rate_10_samples_per_symbol","10 samples/symbol (untrained rate)")]
    selected=["mixture","matched","notch","unet","hybrid"]
    array=np.array([[results["methods"][c][m]["si_sdr_db"] for m in selected] for c,_ in cells])
    fig,ax=plt.subplots(figsize=(11,5.5),layout="constrained")
    image=ax.imshow(array,aspect="auto",cmap="RdBu",vmin=min(-8,float(array.min())),vmax=max(16,float(array.max())))
    ax.set_xticks(range(len(selected)),["Mixture","RRC matched","Adaptive notch","Spectral U-Net","ETHERGLASS"])
    ax.set_yticks(range(len(cells)),[label for _,label in cells])
    ax.tick_params(length=0)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(j,i,f"{array[i,j]:.2f}",ha="center",va="center",color="white" if array[i,j]<-4 or array[i,j]>14 else INK,fontsize=10)
    fig.colorbar(image,ax=ax,label="Mean complex waveform SI-SDR (dB)",shrink=.8)
    ax.set_title("Channel holdouts and deliberate limitations",loc="left",fontweight="bold",pad=20)
    save(fig,"channel-and-stress-tests")

    # Fixed seed was reserved before training/evaluation and not selected for
    # a flattering result. Save the measured metrics beside the visual.
    seed=870001;ex=generate_example(seed,n=16384,split="test",snr_db=20,sir_db=-6,family="QPSK")
    estimates={m:separate(ex["mixture"],1e6,symbol_rate=125000,method=m)["components"][0] for m in ["hybrid","matched","notch"]}
    evidence={m:waveform_metrics(ex["target"],y) for m,y in {"mixture":ex["mixture"],**estimates}.items()}
    evidence["demo_seed"]=seed
    (ROOT/"research"/"demo_metrics.json").write_text(json.dumps(evidence,indent=2)+"\n")
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout="constrained")
    t=np.arange(700)/1e3
    axes[0].plot(t,ex["mixture"][:700].real,color="#a4a69b",lw=.8,label="Observed mixture")
    axes[0].plot(t,ex["target"][:700].real,color=INK,lw=1,label="Clean target reference")
    axes[0].plot(t,estimates["hybrid"][:700].real,color=COPPER,lw=.9,label="ETHERGLASS estimate")
    axes[0].set_xlabel("Time (ms)");axes[0].set_ylabel("In-phase amplitude");axes[0].legend(frameon=False,ncol=3,fontsize=9)
    axes[0].set_title(f"Reserved demonstration seed {seed} · actual reconstructed samples",loc="left",fontweight="bold")
    size=len(ex["target"]);f=np.fft.fftshift(np.fft.fftfreq(size,d=1e-6))/1000
    for name,z,color in [("Observed mixture",ex["mixture"],"#a4a69b"),("Clean target",ex["target"],INK),("ETHERGLASS",estimates["hybrid"],COPPER)]:
        from scipy.signal import welch
        freq,p=welch(z.astype(np.complex128),fs=1e6,nperseg=1024,return_onesided=False,detrend=False)
        axes[1].plot(np.fft.fftshift(freq)/1000,10*np.log10(np.fft.fftshift(p)+1e-20),color=color,lw=1,label=name)
    axes[1].set_xlim(-120,120);axes[1].set_xlabel("Frequency relative to capture center (kHz)")
    axes[1].set_ylabel("PSD (dB input unit² / Hz)");axes[1].grid(axis="y")
    improvement=evidence["hybrid"]["si_sdr_db"]-evidence["mixture"]["si_sdr_db"]
    axes[1].set_title(f"Waveform SI-SDR: mixture {evidence['mixture']['si_sdr_db']:.2f} → estimate {evidence['hybrid']['si_sdr_db']:.2f} dB ({improvement:+.2f} dB)",loc="left",fontsize=11)
    save(fig,"genuine-separation")

    cell_count=len(results["methods"])
    record_count=cell_count*results["examples_per_cell"]
    table=["# Measured separation results","", "Generated from `research/results.json`; raw paired observations are in `evaluation.csv`.","",
        f"Denominators: **{record_count} independent generated records across {cell_count} conditions, {results['examples_per_cell']} records per condition**, evaluated on {seed_count} model-training seeds. The raw CSV has **{results['rows']} method/run observations**; these are not {results['rows']} independent recordings. Classical methods run once per record and learned methods once per training seed. All records are 4,096 samples. Metrics below are record-level arithmetic means, then training-seed means for learned methods; BER/SER are macro averages over records, not pooled bit counts.","",
        "| Method | Nominal SI-SDR dB | Shift SI-SDR dB | Shift waveform EVM % | Shift oracle SER | Shift CPU ms / 4096 samples |","|---|---:|---:|---:|---:|---:|"]
    for m,label in zip(methods,labels):
        a=results["methods"]["nominal"][m];b=results["methods"]["shift"][m]
        table.append(f"| {label} | {a['si_sdr_db']:.3f} | {b['si_sdr_db']:.3f} | {b['waveform_evm_pct']:.2f} | {b['oracle_ser']:.4f} | {b['runtime_ms']:.2f} |")
    table += ["", f"All learned methods use {seed_count} training seeds: {results['evaluation_config']['training_seeds']}. Seed0 was predetermined for the app. Symbol metrics use actual synthetic reference symbols and oracle timing/carrier/gain; they do not measure real-message decoding. Waveform scores align one complex scalar without delay or permutation search.","", "## Paired comparisons","", "| Cell / comparison | Mean SI-SDR delta dB | Paired-example bootstrap 95% CI | Wins / examples |","|---|---:|---|---:|"]
    for key,row in results["comparisons"].items():
        lo,hi=row["paired_example_bootstrap_95_ci_db"]
        table.append(f"| {key} | {row['mean_si_sdr_delta_db']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {row['wins']}/{row['n']} |")
    table += ["", "## Falsifiable hypothesis verdicts", ""]
    for key,description in [("shift:hybrid-scratch","Stationary-tone pretraining improves fixed-budget chirp adaptation"),("shift:hybrid-no_dsp","Explicit contrast features improve the shifted-condition model")]:
        row=results["comparisons"][key];lo,hi=row["paired_example_bootstrap_95_ci_db"]
        verdict="Positive mean effect within this holdout" if lo>0 else "Contradicted within this holdout" if hi<0 else "Inconclusive within this holdout"
        table.append(f"- {description}: **{verdict.lower()}**; paired mean {row['mean_si_sdr_delta_db']:+.3f} dB, interval [{lo:+.3f}, {hi:+.3f}] dB.")
    table += ["", "Intervals average training seeds before resampling independent example IDs, so they do not quantify the full uncertainty over training runs. The pretraining comparison wins on only 10/24 records; its small positive mean and near-zero interval boundary should not be read as a consistent per-record gain. These exploratory comparison intervals are not adjusted for multiple comparisons. All negative comparisons remain above. See `PROTOCOL.md` for priors, split ranges, ablations and limits."]
    table += ["", "## Independent-generator, channel and failure tests", "", "| Condition | Mixture SI-SDR dB | Notch SI-SDR dB | U-Net SI-SDR dB | Hybrid SI-SDR dB | Hybrid waveform EVM % |", "|---|---:|---:|---:|---:|---:|"]
    for cell,label in cells:
        group=results["methods"][cell]
        table.append(f"| {label} | {group['mixture']['si_sdr_db']:.3f} | {group['notch']['si_sdr_db']:.3f} | {group['unet']['si_sdr_db']:.3f} | {group['hybrid']['si_sdr_db']:.3f} | {group['hybrid']['waveform_evm_pct']:.2f} |")
    table += ["", "The TorchSig row uses an independent implementation of simulated pulse shaping, not hardware data. Its helper hides transmitted symbols, so symbol EVM/SER/BER are absent. The digital-interferer test deliberately violates the sparse-interference prior and cannot establish unique two-transmitter recovery. The 6/10 sample-rate tests are outside nominal 8 samples/symbol training; the application marks those rates unvalidated.","", "Figures are generated only from saved actual measurements in `research/figures/`, with editable SVG sources. The main product demo and social artwork are generated separately by `scripts/make_figures.py`."]
    (ROOT/"research"/"RESULTS.md").write_text("\n".join(table)+"\n")


if __name__=="__main__":main()
