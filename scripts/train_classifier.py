#!/usr/bin/env python3
"""Train and calibrate the portable, conservative PSK feature classifier.

All train, probability-calibration, threshold-selection, and test seeds are
separate. No real-recording labels or test cases are used to choose thresholds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from etherglass.classification import FEATURE_NAMES, extract_features, _sigmoid
from etherglass.synthetic import generate_psk, rrc_taps

UNKNOWN_TRAIN = ["tone", "chirp", "noise", "FSK", "16QAM", "8PSK", "OFDM", "PSK_mixture"]
UNKNOWN_TEST = UNKNOWN_TRAIN + ["AM", "32QAM"]


def _shape(symbols, n, sps, rng):
    h = rrc_taps(sps, rolloff=rng.uniform(.25,.45))
    z = np.zeros(len(symbols)*sps, dtype=complex)
    z[::sps] = symbols
    z = fftconvolve(z, h, mode="same")
    locations = np.arange(n) + len(h) + 2*sps + rng.uniform(0,sps)
    return np.interp(locations,np.arange(len(z)),z.real) + 1j*np.interp(locations,np.arange(len(z)),z.imag)


def example(seed: int, known: bool, index: int, heldout=False, n=2048):
    rng = np.random.default_rng(seed)
    sps = int(rng.choice([6, 8, 10, 12]))
    sample_rate = 1e6
    symbol_rate = sample_rate / sps
    snr = rng.uniform(8, 30)
    if known:
        label = index % 2
        family = ["BPSK", "QPSK"][label]
        z, _, _ = generate_psk(rng, n, sps=sps, family=family,
                              rolloff=rng.uniform(.25, .45),
                              timing=rng.uniform(0, sps), cfo=0, multipath=False)
    else:
        kinds = UNKNOWN_TEST if heldout else UNKNOWN_TRAIN
        family = kinds[index % len(kinds)]
        label = -1
        t = np.arange(n)
        if family == "tone":
            z = np.exp(2j*np.pi*rng.uniform(-.3,.3)/sps*t)
        elif family == "chirp":
            z = np.exp(2j*np.pi*(rng.uniform(-.4,.4)/sps*t + .5*rng.uniform(.05,.5)/sps*t*t/n))
        elif family == "noise":
            z = rng.normal(size=n) + 1j*rng.normal(size=n)
        elif family == "FSK":
            f = np.repeat(rng.choice([-1,1], int(np.ceil(n/sps))), sps)[:n]*.32/sps
            z = np.exp(2j*np.pi*np.cumsum(f))
        elif family == "OFDM":
            # Independently modulated subcarriers with a cyclic prefix; no protocol payload.
            blocks = []
            while sum(len(b) for b in blocks) < n:
                carriers = np.zeros(128,dtype=complex)
                used = np.r_[np.arange(1,10), np.arange(119,128)]
                carriers[used] = np.exp(.5j*np.pi*rng.integers(0,4,len(used)))
                block = np.fft.ifft(carriers)
                blocks.append(np.r_[block[-16:],block])
            z = np.concatenate(blocks)[:n]
        elif family == "PSK_mixture":
            a, _, _ = generate_psk(rng,n,sps=sps,family="QPSK",rolloff=rng.uniform(.25,.45),timing=rng.uniform(0,sps))
            b, _, _ = generate_psk(rng,n,sps=sps,family="QPSK",rolloff=rng.uniform(.25,.45),timing=rng.uniform(0,sps))
            z = a + rng.uniform(.6,1.4)*b
        else:
            count = int(np.ceil((n+4*len(rrc_taps(sps)))/sps))+5
            if family == "8PSK":
                symbols = np.exp(2j*np.pi*rng.integers(0,8,count)/8)
            elif family == "AM":
                symbols = rng.normal(size=count)
            else:
                levels = [-3,-1,1,3] if family == "16QAM" else [-5,-3,-1,1,3,5]
                if family == "32QAM":
                    constellation = np.asarray([a+1j*b for a in levels for b in levels
                                                if not (abs(a)==5 and abs(b)==5)])
                    symbols = rng.choice(constellation, count)
                else:
                    symbols = rng.choice(levels,count)+1j*rng.choice(levels,count)
            z = _shape(symbols,n,sps,rng)
        z = z * np.exp(1j*rng.uniform(0,2*np.pi))
    # Same carrier/channel impairment distribution for every known/unknown family.
    # The classifier cannot use source-specific channel quality as a class proxy.
    if rng.random() < .25:
        taps = np.zeros(6,dtype=complex)
        taps[0] = 1
        taps[2] = rng.uniform(.1,.35)*np.exp(1j*rng.uniform(-np.pi,np.pi))
        taps[5] = rng.uniform(.03,.15)*np.exp(1j*rng.uniform(-np.pi,np.pi))
        z = fftconvolve(z,taps,mode="full")[:n]
    z = z * np.exp(2j*np.pi*rng.uniform(-.015,.015)/sps*np.arange(n))
    z = z / (np.sqrt(np.mean(abs(z)**2))+1e-15)
    noise = (rng.normal(size=n)+1j*rng.normal(size=n))/np.sqrt(2)
    z += noise * 10**(-snr/20)
    return z.astype(np.complex64), sample_rate, symbol_rate, label, family


def dataset(base, count, known, heldout=False):
    features, labels, records = [], [], []
    for i in range(count):
        seed = base+i
        z, fs, sr, label, family = example(seed,known,i,heldout=heldout)
        features.append(extract_features(z,fs,sr))
        labels.append(label)
        records.append({"seed":seed,"family":family,"known":known})
    return np.asarray(features),np.asarray(labels),records


def ece(labels,p,bins=10):
    total = len(labels)
    value = 0.
    for lo in np.linspace(0,1,bins,endpoint=False):
        subset = (p>=lo)&((p<lo+1/bins) if lo < 1-1/bins else (p<=1))
        if subset.any():
            value += subset.sum()/total*abs(p[subset].mean()-labels[subset].mean())
    return float(value)


def wilson(success,n):
    z=1.96
    center=(success/n+z*z/(2*n))/(1+z*z/n)
    radius=z/(1+z*z/n)*np.sqrt(success/n*(1-success/n)/n+z*z/(4*n*n))
    return [float(max(0,center-radius)),float(min(1,center+radius))]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",type=int,default=2000,help="Count per known/unknown group")
    parser.add_argument("--calibration",type=int,default=800)
    parser.add_argument("--test",type=int,default=1000)
    args=parser.parse_args()
    if not (20 <= args.train <= 50000 and 20 <= args.calibration <= 50000 and 10 <= args.test <= 50000):
        parser.error("Counts must fit disjoint seed ranges: train/calibration 20–50000, test 10–50000")
    np.random.seed(71)
    print("Generating independent training/calibration/test feature sets",flush=True)
    train_k,labels,_=dataset(100000,args.train,True)
    train_u,_,_=dataset(200000,args.train,False)
    cal_k,cal_labels,_=dataset(300000,args.calibration,True)
    cal_u,_,_=dataset(400000,args.calibration,False)
    test_k,test_labels,known_records=dataset(1100000,args.test,True,True)
    test_u,_,unknown_records=dataset(1200000,args.test,False,True)
    scaler=StandardScaler().fit(np.r_[train_k,train_u])
    transform=scaler.transform
    classifier=LogisticRegression(C=1,max_iter=1000,random_state=71).fit(transform(train_k),labels)
    support=LogisticRegression(C=1,max_iter=1000,random_state=71).fit(
        transform(np.r_[train_k,train_u]),np.r_[np.ones(args.train),np.zeros(args.train)])
    # First half calibrates probability; second half chooses a decision threshold.
    cut=args.calibration//2
    logits=classifier.decision_function(transform(cal_k[:cut]))
    objective=lambda t: log_loss(cal_labels[:cut],_sigmoid(logits/t),labels=[0,1])
    temperature=float(minimize_scalar(objective,bounds=(.1,10),method="bounded").x)
    support_logits=support.decision_function(transform(np.r_[cal_k[:cut],cal_u[:cut]]))[:,None]
    support_calibrator=LogisticRegression(C=100,max_iter=1000).fit(
        support_logits,np.r_[np.ones(cut),np.zeros(cut)])
    slope=float(support_calibrator.coef_[0,0])
    intercept=float(support_calibrator.intercept_[0])
    calibrated=lambda f: _sigmoid(slope*support.decision_function(transform(f))+intercept)
    # A finite-sample operating point, not a distribution-free unknown guarantee.
    scores_u=calibrated(cal_u[cut:])
    threshold=float(max(.5, np.nextafter(np.quantile(scores_u,.99,method="higher"), np.inf)))
    class_threshold=.8
    model={
        "schema_version":1,"feature_names":FEATURE_NAMES,
        "mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),
        "class_coef":classifier.coef_[0].tolist(),"class_intercept":float(classifier.intercept_[0]),
        "class_temperature":temperature,
        "support_coef":support.coef_[0].tolist(),"support_intercept":float(support.intercept_[0]),
        "support_calibration_slope":slope,"support_calibration_intercept":intercept,
        "support_threshold":threshold,"confidence_threshold":class_threshold,
        "training": {"generator":"etherglass.synthetic.generate_psk plus documented unknown probes",
            "seed":71,"train_known_seed_start":100000,"train_unknown_seed_start":200000,
            "calibration_known_seed_start":300000,"calibration_unknown_seed_start":400000,
            "test_known_seed_start":1100000,"test_unknown_seed_start":1200000,
            "exploratory_evaluation_seeds_excluded_from_final":[500000,600000,700000,800000],
            "n_train_per_group":args.train,"n_calibration_per_group":args.calibration,
            "n_test_per_group":args.test,"samples_per_example":2048,
            "snr_db":[8,30],"sps":[6,8,10,12],"rolloff":[.25,.45],
            "common_channel_policy":"Same CFO, 25% multipath, SNR and RMS distributions for every family",
            "threshold_policy":"max(.5, 99th percentile calibration unknown score), class confidence >= .8",
            "unknown_train_families":UNKNOWN_TRAIN,"unknown_test_families":UNKNOWN_TEST,
            "test_only_families":["AM","32QAM"],"no_recording_training":True},
        "limitations":["Finite synthetic probes do not establish open-world RF recognition.",
            "Confidence is conditional on BPSK/QPSK; support probability depends on the balanced probe calibration prior.",
            "Symbol-rate prior is supplied by caller; centered approximately isolated signal is required.",
            "Real RF recordings are unlabeled in this release, so real recognition accuracy is unestablished.",
            "Inference uses the first 2048 candidate samples; only 6/8/10/12 samples-per-symbol priors were calibrated."],
    }
    feature_path = ROOT/"models"/"classifier_features.npz"
    np.savez_compressed(feature_path, train_known=train_k, train_unknown=train_u,
                        train_labels=labels, calibration_known=cal_k, calibration_unknown=cal_u,
                        calibration_labels=cal_labels, test_known=test_k, test_unknown=test_u,
                        test_labels=test_labels)
    model["training"]["features_sha256"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    model["training"]["source_sha256"] = {
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT/"src"/"etherglass"/"classification.py", ROOT/"src"/"etherglass"/"synthetic.py"]}
    path=ROOT/"models"/"classifier.json"
    path.write_text(json.dumps(model,indent=2)+"\n")
    p=_sigmoid(classifier.decision_function(transform(test_k))/temperature)
    confidence=np.maximum(p,1-p)
    pred=(p>=.5).astype(int)
    support_known=calibrated(test_k)
    support_unknown=calibrated(test_u)
    ku=(support_known>=threshold)&(confidence>=class_threshold)
    unknown_p=_sigmoid(classifier.decision_function(transform(test_u))/temperature)
    unknown_confidence=np.maximum(unknown_p,1-unknown_p)
    uu=(support_unknown>=threshold)&(unknown_confidence>=class_threshold)
    metrics={
        "known_class_accuracy_before_abstention":float(accuracy_score(test_labels,pred)),
        "known_coverage":float(ku.mean()),
        "known_accepted_accuracy":float(accuracy_score(test_labels[ku],pred[ku])) if ku.any() else None,
        "unknown_false_accept_rate":float(uu.mean()),
        "unknown_false_accept_wilson95":wilson(int(uu.sum()),len(uu)),
        "unknown_detection_auroc":float(roc_auc_score(np.r_[np.ones(len(test_k)),np.zeros(len(test_u))],np.r_[support_known,support_unknown])),
        "known_brier_score":float(brier_score_loss(test_labels,p)),
        "known_support_brier_score":float(brier_score_loss(np.r_[np.ones(len(test_k)),np.zeros(len(test_u))],np.r_[support_known,support_unknown])),
        "known_support_binary_ece_10_bins":ece(np.r_[np.ones(len(test_k)),np.zeros(len(test_u))],np.r_[support_known,support_unknown]),
        "known_binary_ece_10_bins":ece(test_labels,p),
        "known_test_n":len(test_k),"unknown_test_n":len(test_u),
        "support_threshold":threshold,"confidence_threshold":class_threshold,
        "threshold_selected_without_test":True,
        "unknown_by_family":{},
        "model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for family in UNKNOWN_TEST:
        mask=np.asarray([v["family"]==family for v in unknown_records])
        metrics["unknown_by_family"][family]={"n":int(mask.sum()),"false_accept_rate":float(uu[mask].mean())}
    rows=[]
    for i,r in enumerate(known_records):
        rows.append({**r,"true_class":["BPSK","QPSK"][test_labels[i]],"predicted_class":["BPSK","QPSK"][pred[i]],
                     "class_confidence":float(confidence[i]),"support_score":float(support_known[i]),"accepted":bool(ku[i])})
    for i,r in enumerate(unknown_records):
        rows.append({**r,"true_class":"unknown","predicted_class":"QPSK" if unknown_p[i]>=.5 else "BPSK",
                     "class_confidence":float(unknown_confidence[i]),"support_score":float(support_unknown[i]),"accepted":bool(uu[i])})
    with (ROOT/"models"/"classifier_evaluation.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (ROOT/"models"/"classifier_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=="__main__":
    main()
