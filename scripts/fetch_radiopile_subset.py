#!/usr/bin/env python3
"""Fetch a 40-window CC-BY-4.0 RadioPile ITS excerpt with a 4 MiB network cap.

Requires NumPy and h5py. Downloads via public HTTP Range only; refuses servers
that ignore Range. Does not download or verify the full 388 MB source artifact.
The output manifest preserves original observation IDs, attribution, transforms,
and an actual SHA256 of exported bytes. Neighboring windows are correlated.
"""
import io
import json
import time
from urllib.request import Request, urlopen

class BoundedHTTP(io.RawIOBase):
    def __init__(self, url, size=388327096, budget=4*1024*1024, block=65536):
        self.url, self.size, self.budget, self.block = url, size, budget, block
        self.pos, self.downloaded, self.cache = 0, 0, {}
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos
    def readinto(self, buffer):
        result = self.read(len(buffer))
        buffer[:len(result)] = result
        return len(result)
    def read(self, n=-1):
        if n < 0: n = self.size-self.pos
        end = min(self.pos+n, self.size)
        output = bytearray()
        while self.pos < end:
            index = self.pos//self.block
            if index not in self.cache:
                start, stop = index*self.block, min((index+1)*self.block,self.size)-1
                if self.downloaded+stop-start+1 > self.budget:
                    raise RuntimeError('Public-data range download budget exceeded')
                request = Request(self.url, headers={'Range':f'bytes={start}-{stop}', 'User-Agent':'ETHERGLASS-research/0.1'})
                with urlopen(request, timeout=30) as response:
                    if response.status != 206:
                        raise RuntimeError('Range request was not honored; refusing whole-file transfer')
                    data = response.read(self.block+1)
                    if len(data) != stop-start+1: raise RuntimeError('Unexpected byte-range length')
                self.cache[index] = data
                self.downloaded += len(data)
                print(f'public source bytes fetched: {self.downloaded}', flush=True)
            chunk = self.cache[index]
            within = self.pos-index*self.block
            amount = min(end-self.pos,len(chunk)-within)
            output.extend(chunk[within:within+amount])
            self.pos += amount
        return bytes(output)


import hashlib
from pathlib import Path
import h5py
import numpy as np

source=BoundedHTTP('https://zenodo.org/records/22877108/files/its_band.h5?download=1')
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'examples'/'real')
out = parser.parse_args().output
out.mkdir(parents=True,exist_ok=True)
def text(x): return x.decode() if isinstance(x,bytes) else str(x)
with h5py.File(source,'r') as f:
    names=[text(x) for x in f['label_names'][:]]
    print('label names',names,flush=True)
    rows=[]
    arrays=[]
    for start in [0,15000,30000,45000,60000]:
        stop=start+8
        data=f['data'][start:stop]
        labels=f['labels'][start:stop]
        for j in range(8):
            index=start+j
            row={'window_index':index,'technology':names[int(labels[j])],
                 'modulation':'unknown','sample_rate_hz':25000000.0,'center_frequency_hz':5900000000.0}
            for key in ['sample_index','recording_id','file_path']:
                value=f['meta'][key][index]
                row[key]=int(value) if key=='sample_index' else text(value)
            rows.append(row)
            arrays.append(data[j])
        print('extracted source rows',start,stop,flush=True)
    attrs={str(k):text(v) for k,v in f.attrs.items()}
array=np.asarray(arrays,dtype=np.float32)
path=out/'radiopile-its-40.npz'
np.savez_compressed(path,iq=array,window_indices=np.array([r['window_index'] for r in rows]),technology=np.array([r['technology'] for r in rows]))
manifest={'source_doi':'10.5281/zenodo.22877108','source_url':source.url,
    'original_filename':'its_band.h5','original_size_bytes':388327096,
    'original_md5_published':'8eff5d7867ea4823901866852d05b5f2','original_full_file_checksum_verified':False,
    'license':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
    'attribution':'M. Girmay et al., Vehicular Communications 39 (2023),100563; processed by Xander Vanparys et al., RadioPile v1.0.',
    'source_paper_doi':'10.1016/j.vehcom.2022.100563',
    'modifications':'Selected 8 consecutive windows at each of the first five 15000-window class blocks; converted selected float32 rows to NPZ. Source already unit-power normalized and augmented with synthetic AWGN. No additional denoising or augmentation.',
    'use':'Independent external unsupported-input/stability test only. Not used for model training or threshold selection.',
    'limitations':['No clean waveform or reference symbols. No ground-truth separation quality, BER or EVM available.','Neighboring windows can share an observation. file_path/recording_id do not prove independent device/session identities.','40 selected windows are a convenience subset, not a representative sample of the full dataset.'],
    'fetched_bytes':source.downloaded,'range_budget_bytes':source.budget,
    'subset_file':path.name,'subset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'shape':list(array.shape),'dtype':'float32','rows':rows,'original_hdf5_attributes':attrs}
(out/'radiopile-its-40.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'downloaded':source.downloaded,'shape':array.shape,'sha256':manifest['subset_sha256']}),flush=True)
