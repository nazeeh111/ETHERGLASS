#!/usr/bin/env python3
"""Reproduce training, held-out evaluation, ablations and raw evidence on CPU."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from etherglass.synthetic import generate_example,generate_psk,rrc_taps
from etherglass.models import FRAME_SIZE,make_model,spectral_features,CPUConv1d
from etherglass.separation import separate
from etherglass.metrics import waveform_metrics,synthetic_symbol_metrics

CONFIG={"frame_size":2048,"samples_per_symbol":8,"rolloff":.35,
    "training_examples":1536,"adaptation_examples":512,"pretrain_steps":320,
    "adaptation_steps":160,"batch_size":24,"learning_rate":.002,
    "adapt_learning_rate":.0008,"train_seed_start":100000,
    "adapt_seed_start":200000,"validation_seed_start":300000,
    "test_seed_start":400000,"demo_seed":870001,"training_seeds":[0,1,2],
    "scope":"Known-rate PSK target + sparse tones/chirps; no blind cochannel source recovery",
    "hypothesis":"At a fixed 160-example-update adaptation budget, pretraining on stationary tones followed by chirp adaptation reduces held-out chirp waveform error relative to the identical network initialized from scratch. Local multiscale spectral contrasts improve robustness relative to their feature ablation.",
    "selection":"Fixed step budget, no test selection; released seed 0 is predetermined.",
    "loss":"Complex frequency MSE normalized by per-example clean-target energy, Hann-windowed frames.",
    "metric_alignment":"No delay/permutation search. SI-SDR complex projection; EVM is waveform EVM after one scalar gain/phase alignment, NOT symbol EVM/BER.",
    "baseline_priors":"All methods know target center/rate. U-Net has identical 7 features and support prior; classical methods use rate/support. No method sees clean truth at inference."}


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def dataset(start,count,split):
    features=[]; spectra=[]; targets=[]
    win=np.hanning(FRAME_SIZE).astype(np.float32)
    for i in range(count):
        ex=generate_example(start+i,n=FRAME_SIZE,split=split)
        f,s,scale=spectral_features(ex["mixture"]*win)
        clean=np.fft.fftshift(np.fft.fft(ex["target"]*win/scale,norm="ortho"))
        features.append(f);spectra.append(s);targets.append(clean)
    return torch.from_numpy(np.stack(features)),torch.from_numpy(np.stack(spectra)),torch.from_numpy(np.stack(targets).astype(np.complex64))


def fit(model,data,steps,seed,lr,stage,log):
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    model.train(); t=time.perf_counter()
    f,x,y=data
    for step in range(steps):
        idx=torch.from_numpy(rng.integers(0,len(f),CONFIG["batch_size"]))
        gain=model(f[idx]);pred=gain*x[idx]
        loss=((abs(pred-y[idx])**2).mean(-1)/(abs(y[idx])**2).mean(-1).clamp_min(1e-7)).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if step%40==0 or step==steps-1:
            row={"stage":stage,"step":step+1,"loss":float(loss.detach()),"elapsed_s":time.perf_counter()-t}
            log.append(row);print(json.dumps(row),flush=True)
    return time.perf_counter()-t


def checkpoint(path,model,architecture,seed,stages,seconds):
    obj={"state_dict":model.state_dict(),"architecture":architecture,"training_seed":seed,
        "parameter_count":sum(p.numel() for p in model.parameters()),"stages":stages,
        "frame_size":FRAME_SIZE,"samples_per_symbol":8,"rolloff":.35,
        "training_seconds":seconds,"version":"0.1.0","license":"MIT (weights trained exclusively on CC0 synthetic data)",
        "scope":CONFIG["scope"]}
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,temporary)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_checkpoint(source,destination):
    temporary=destination.with_suffix(destination.suffix+".tmp")
    shutil.copyfile(source,temporary)
    temporary.replace(destination)


def train(seeds,quick=False,resume=False):
    torch.set_num_threads(4)
    if quick:
        CONFIG.update(training_examples=512,adaptation_examples=192,pretrain_steps=100,adaptation_steps=60)
    config_path=ROOT/"research"/"config.json"
    if resume and config_path.exists():
        saved=json.loads(config_path.read_text())
        if quick and (saved["pretrain_steps"]!=100 or saved["adaptation_steps"]!=60):
            raise ValueError("--quick cannot alter a saved full training run; use a separate checkout")
        CONFIG.clear();CONFIG.update(saved)
    else:
        write_json(config_path,CONFIG)
    runtime_path=ROOT/"research"/"training_runtime.json"
    runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    for seed in seeds:
        runtime.setdefault(str(seed),{"torch":torch.__version__,"numpy":np.__version__,"threads":4})
        runtime[str(seed)].setdefault("architecture_backends",{})
    write_json(runtime_path,runtime)
    print("Generating physically labelled train/adaptation datasets",flush=True)
    train_data=dataset(CONFIG["train_seed_start"],CONFIG["training_examples"],"train")
    adapt_data=dataset(CONFIG["adapt_seed_start"],CONFIG["adaptation_examples"],"adapt")
    manifest=[]
    for seed in seeds:
        for arch in ["hybrid","no_dsp","unet"]:
            existing_path=ROOT/"models"/f"seed{seed}"/f"{arch}.pt"
            log_path=ROOT/"research"/"training_logs"/f"{arch}_seed{seed}.json"
            if resume and existing_path.is_file() and log_path.is_file():
                logs=json.loads(log_path.read_text())
                for stage,expected in [("pretrain",CONFIG["pretrain_steps"]),("adapt",CONFIG["adaptation_steps"])]:
                    selected=[r for r in logs if f"/{stage}/" in r["stage"]]
                    if not selected or selected[-1]["step"]!=expected:
                        raise ValueError(f"cannot resume inconsistent completed stage: {arch}/seed{seed}/{stage}")
                obj=torch.load(existing_path,map_location="cpu",weights_only=True)
                if obj["training_seed"]!=seed or obj["architecture"]!=arch:
                    raise ValueError(f"cannot resume checkpoint identity mismatch: {existing_path}")
                runtime[str(seed)]["architecture_backends"].setdefault(arch,{
                    "convolution":runtime[str(seed)].get("convolution","previous completed run"),
                    "floating_point":runtime[str(seed)].get("floating_point","previous completed run")})
                write_json(runtime_path,runtime)
                print(f"Preserving completed checkpoint bytes: {existing_path}",flush=True)
                manifest.append({"file":str(existing_path.relative_to(ROOT)),"sha256":hashlib.sha256(existing_path.read_bytes()).hexdigest()})
                if arch=="hybrid":
                    p=existing_path.with_name("pretrained.pt")
                    if not p.is_file(): raise FileNotFoundError("completed hybrid requires its original pretrained checkpoint")
                    manifest.append({"file":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
                continue
            runtime[str(seed)]["architecture_backends"][arch]={"convolution":"CPU unfold+GEMM (same kernels/weights)" if CPUConv1d.use_gemm else "native torch.nn.Conv1d",
                "floating_point":"float32 weights/features; complex128 RMS accumulation before normalization"}
            write_json(runtime_path,runtime)
            torch.manual_seed(seed);model=make_model(arch);logs=[]
            sec=fit(model,train_data,CONFIG["pretrain_steps"],seed,CONFIG["learning_rate"],f"{arch}/pretrain/seed{seed}",logs)
            if arch=="hybrid":
                p=ROOT/"models"/f"seed{seed}"/"pretrained.pt"
                sha=checkpoint(p,model,arch,seed,["stationary-tone pretraining"],sec)
                manifest.append({"file":str(p.relative_to(ROOT)),"sha256":sha})
                if seed==0: copy_checkpoint(p,ROOT/"models"/"pretrained.pt")
            sec+=fit(model,adapt_data,CONFIG["adaptation_steps"],seed+100,CONFIG["adapt_learning_rate"],f"{arch}/adapt/seed{seed}",logs)
            p=ROOT/"models"/f"seed{seed}"/f"{arch}.pt"
            sha=checkpoint(p,model,arch,seed,["stationary-tone pretraining","chirp/channel adaptation"],sec)
            manifest.append({"file":str(p.relative_to(ROOT)),"sha256":sha})
            if seed==0: copy_checkpoint(p,ROOT/"models"/f"{arch}.pt")
            write_json(ROOT/"research"/"training_logs"/f"{arch}_seed{seed}.json",logs)
        scratch_path=ROOT/"models"/f"seed{seed}"/"scratch.pt"
        if resume and scratch_path.is_file():
            log_path=ROOT/"research"/"training_logs"/f"scratch_seed{seed}.json"
            logs=json.loads(log_path.read_text())
            if not logs or logs[-1]["step"]!=CONFIG["adaptation_steps"]: raise ValueError("scratch log/config mismatch")
            manifest.append({"file":str(scratch_path.relative_to(ROOT)),"sha256":hashlib.sha256(scratch_path.read_bytes()).hexdigest()})
            print(f"Preserving completed checkpoint bytes: {scratch_path}",flush=True)
            continue
        runtime[str(seed)]["architecture_backends"]["scratch"]={"convolution":"CPU unfold+GEMM (same kernels/weights)" if CPUConv1d.use_gemm else "native torch.nn.Conv1d",
            "floating_point":"float32 weights/features; complex128 RMS accumulation before normalization"}
        write_json(runtime_path,runtime)
        torch.manual_seed(seed);model=make_model("scratch");logs=[]
        sec=fit(model,adapt_data,CONFIG["adaptation_steps"],seed+100,CONFIG["adapt_learning_rate"],f"scratch/adapt/seed{seed}",logs)
        p=ROOT/"models"/f"seed{seed}"/"scratch.pt"
        sha=checkpoint(p,model,"hybrid",seed,["chirp/channel adaptation from random initialization"],sec)
        manifest.append({"file":str(p.relative_to(ROOT)),"sha256":sha})
        if seed==0: copy_checkpoint(p,ROOT/"models"/"scratch.pt")
        write_json(ROOT/"research"/"training_logs"/f"scratch_seed{seed}.json",logs)
    manifest_path=ROOT/"research"/"checkpoint_manifest.json"
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    merged={r["file"]:r for r in previous+manifest}
    write_json(manifest_path,list(merged.values()))


def conditions(count):
    cells=[("nominal",{"split":"nominal"}), ("shift",{"split":"test"})]
    for snr in [0,10,20,30]: cells.append((f"snr_{snr}",{"split":"test","snr_db":snr}))
    for sir in [-15,-5,5,15]: cells.append((f"sir_{sir}",{"split":"test","sir_db":sir}))
    for impairment in ["flat","multipath","timing","carrier"]:
        cells.append((impairment,{"split":"test","impairment":impairment}))
    for j,(label,kwargs) in enumerate(cells):
        for i in range(count):
            seed=CONFIG["test_seed_start"]+j*1000+i
            yield label,seed,generate_example(seed,n=4096,**kwargs)
    # A deliberately outside-scope stress test: arbitrary overlapping digital
    # sources are not identifiable from a single sensor without extra priors.
    for i in range(count):
        seed=CONFIG["test_seed_start"]+20000+i
        ex=generate_example(seed,n=4096,split="test",sir_db=-5)
        rng=np.random.default_rng(seed+900000)
        interferer,_,_=generate_psk(rng,4096,sps=8,family="QPSK",cfo=.004,timing=.5)
        ex["interference"]=interferer*10**(.25)
        ex["mixture"]=(ex["target"]+ex["interference"]+ex["noise"]).astype(np.complex64)
        ex["metadata"]["interference"]={"kind":"overlapping_QPSK_stress"}
        yield "overlapping_digital_stress",seed,ex
    from etherglass._vendor.torchsig_psk import constellation_modulator_baseband
    for i in range(count):
        seed=CONFIG["test_seed_start"]+30000+i
        ex=generate_example(seed,n=4096,split="nominal")
        rng=np.random.default_rng(seed+900000)
        target=constellation_modulator_baseband(constellation_name=ex["metadata"]["family"].lower(),
            pulse_shape_name="srrc",max_num_samples=8192,oversampling_rate_nominal=8,
            alpha_rolloff=.35,rng=rng)[2048:6144]
        target=target/(np.sqrt(np.mean(abs(target)**2))+1e-12)
        ex["target"]=target.astype(np.complex64)
        ex["mixture"]=(ex["target"]+ex["interference"]+ex["noise"]).astype(np.complex64)
        ex["symbols"]=None  # Upstream helper does not expose transmitted symbols.
        ex["metadata"]["generator"]="TorchSig/1197261d4f0d4c93de18757ee2ceb368ac04104c extracted upstream helper"
        ex["metadata"].update(carrier_offset_cycles_per_sample=0.,phase_rad=0.,
            channel_taps=[[1.,0.]],symbol_labels=None,symbol_crop_offset_samples=None,
            timing_samples=None)
        yield "torchsig_independent_generator",seed,ex
    for sps in [6,10]:
        for i in range(count):
            seed=CONFIG["test_seed_start"]+40000+sps*1000+i
            ex=generate_example(seed,n=4096,split="test")
            rng=np.random.default_rng(seed+900000)
            target,symbols,meta=generate_psk(rng,4096,sps=sps,family=ex["metadata"]["family"],multipath=True)
            ex["target"]=target;ex["symbols"]=symbols;ex["metadata"].update(meta)
            ex["metadata"]["symbol_rate"]=1e6/sps
            ex["mixture"]=(target+ex["interference"]+ex["noise"]).astype(np.complex64)
            yield f"rate_{sps}_samples_per_symbol",seed,ex


def evaluate(seeds,count=24):
    # Evaluation must describe the run actually trained, including --quick.
    # Never silently substitute this source file's default training budgets.
    config_path=ROOT/"research"/"config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Train first: saved research/config.json is required for evaluation")
    saved_config=json.loads(config_path.read_text())
    required={"frame_size","samples_per_symbol","training_examples","adaptation_examples","pretrain_steps","adaptation_steps","train_seed_start","adapt_seed_start","test_seed_start"}
    if not required.issubset(saved_config):
        raise ValueError(f"saved training config lacks fields: {required-set(saved_config)}")
    CONFIG.clear();CONFIG.update(saved_config)
    evaluated_checkpoints=[]
    manifest_path=ROOT/"research"/"checkpoint_manifest.json"
    if not manifest_path.is_file(): raise FileNotFoundError("completed training checkpoint manifest is required")
    manifest={row["file"]:row["sha256"] for row in json.loads(manifest_path.read_text())}
    for seed in seeds:
        for method in ["hybrid","pretrained","no_dsp","scratch","unet"]:
            path=ROOT/"models"/f"seed{seed}"/f"{method}.pt"
            key=str(path.relative_to(ROOT));digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if manifest.get(key)!=digest: raise ValueError(f"checkpoint manifest mismatch: {key}")
            obj=torch.load(path,map_location="cpu",weights_only=True)
            if obj["training_seed"]!=seed or obj["frame_size"]!=CONFIG["frame_size"] or obj["samples_per_symbol"]!=CONFIG["samples_per_symbol"]:
                raise ValueError(f"checkpoint/config mismatch: {key}")
            log_method="hybrid" if method=="pretrained" else method
            logs=json.loads((ROOT/"research"/"training_logs"/f"{log_method}_seed{seed}.json").read_text())
            stages=["adapt"] if method=="scratch" else ["pretrain"] if method=="pretrained" else ["pretrain","adapt"]
            for stage in stages:
                rows=[r for r in logs if f"/{stage}/" in r["stage"]]
                expected=CONFIG["pretrain_steps"] if stage=="pretrain" else CONFIG["adaptation_steps"]
                if not rows or rows[-1]["step"]!=expected: raise ValueError(f"training log/config step mismatch: {key}/{stage}")
            evaluated_checkpoints.append({"file":key,"sha256":digest,"parameter_count":obj["parameter_count"]})
    torch.set_num_threads(4)
    methods=["mixture","bandpass","matched","wiener","notch","pretrained","hybrid","no_dsp","scratch","unet"]
    rows=[]
    for cell,example_seed,ex in conditions(count):
        for method in methods:
            selected_seeds=seeds if method in {"hybrid","unet","pretrained","no_dsp","scratch"} else [-1]
            for seed in selected_seeds:
                if method=="mixture": y=ex["mixture"]; runtime=0.
                else:
                    cp=ROOT/"models"/f"seed{seed}"/f"{method}.pt" if seed>=0 else None
                    result=separate(ex["mixture"],1e6,symbol_rate=ex["metadata"]["symbol_rate"],method=method,checkpoint=cp)
                    y=result["components"][0];runtime=result["diagnostics"]["runtime_ms"]
                metrics=waveform_metrics(ex["target"],y)
                if ex["symbols"] is not None:
                    metrics.update(synthetic_symbol_metrics(y,ex["symbols"],ex["metadata"],already_matched=method=="matched"))
                else:
                    metrics.update(oracle_symbol_evm_pct=None,oracle_ser=None,oracle_ber=None,reference_symbol_count=0)
                rows.append({"cell":cell,"example_seed":example_seed,"training_seed":seed,"method":method,
                             "snr_db":ex["metadata"]["snr_db"],"sir_db":ex["metadata"]["sir_db"],
                             "family":ex["metadata"]["family"],
                             "carrier_offset_cycles_per_sample":ex["metadata"]["carrier_offset_cycles_per_sample"],
                             "timing_samples":ex["metadata"]["timing_samples"],
                             "channel_taps_json":json.dumps(ex["metadata"]["channel_taps"]),
                             "generator":ex["metadata"]["generator"],
                             "runtime_ms":runtime,**metrics})
        if example_seed%1000==0: print(f"evaluated {cell}",flush=True)
    raw=ROOT/"research"/"evaluation.csv"
    with raw.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"config":dict(CONFIG),"rows":len(rows),"examples_per_cell":count,"methods":{},
        "evaluation_config":{"training_seeds":seeds,"examples_per_cell":count,"record_samples":4096,
            "inference_api":"etherglass.separation.separate","threads":4,
            "training_config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "evaluated_checkpoints":evaluated_checkpoints}
    for cell in sorted({r["cell"] for r in rows}):
        summary["methods"][cell]={}
        for method in methods:
            subset=[r for r in rows if r["cell"]==cell and r["method"]==method]
            summary["methods"][cell][method]={key:float(np.mean(values)) if (values:=[r[key] for r in subset if r[key] is not None]) else None for key in ["si_sdr_db","waveform_evm_pct","nmse_db","runtime_ms","oracle_symbol_evm_pct","oracle_ser","oracle_ber"]}
            summary["methods"][cell][method]["training_seed_mean_si_sdr_db"]={str(s):float(np.mean([r["si_sdr_db"] for r in subset if r["training_seed"]==s])) for s in sorted({r["training_seed"] for r in subset})}
    comparisons={}
    for a,b in [("hybrid","scratch"),("hybrid","pretrained"),("hybrid","no_dsp"),("hybrid","unet"),("hybrid","wiener"),("hybrid","notch"),("hybrid","matched")]:
        for cell in ["nominal","shift"]:
            ids=sorted({r["example_seed"] for r in rows if r["cell"]==cell})
            delta=[]
            for i in ids:
                vals={m:np.mean([r["si_sdr_db"] for r in rows if r["cell"]==cell and r["example_seed"]==i and r["method"]==m]) for m in [a,b]}
                delta.append(vals[a]-vals[b])
            delta=np.asarray(delta);rng=np.random.default_rng(817)
            boot=np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1)
            comparisons[f"{cell}:{a}-{b}"]={"mean_si_sdr_delta_db":float(delta.mean()),"paired_example_bootstrap_95_ci_db":np.quantile(boot,[.025,.975]).tolist(),"wins":int(np.sum(delta>0)),"n":len(delta),"note":"Examples are resampled after averaging training seeds; does not estimate population training-seed uncertainty."}
    summary["comparisons"]=comparisons
    write_json(ROOT/"research"/"results.json",summary)
    print(json.dumps({"shift":summary["methods"]["shift"],"comparisons":comparisons},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["train","evaluate","all"])
    p.add_argument("--seeds",default="0,1,2");p.add_argument("--count",type=int,default=24)
    p.add_argument("--quick",action="store_true")
    p.add_argument("--native-conv",action="store_true",help="use native PyTorch CPU convolution rather than equivalent GEMM path")
    p.add_argument("--resume",action="store_true",help="preserve verified completed checkpoints/logs; restart only incomplete architectures")
    a=p.parse_args()
    if a.native_conv: CPUConv1d.use_gemm=False
    seeds=[int(s) for s in a.seeds.split(",")]
    if a.action in {"train","all"}: train(seeds,a.quick,a.resume)
    if a.action in {"evaluate","all"}: evaluate(seeds,a.count)
