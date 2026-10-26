#!/usr/bin/env python3
"""Fetch only the first 65,536 complex samples of a CC-BY CAMRAS observation."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

BASE='https://data.camras.nl/satellites/raw/'
NAME='camras-2022_08_07_08_31_34_2260.830MHz_0.5Msps_ci16_le'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'examples'/'real')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with urlopen(BASE+NAME+'.sigmf-meta',timeout=30) as response:
        metadata_bytes=response.read(65537)
    if len(metadata_bytes)>65536: raise RuntimeError('Unexpected metadata size')
    original=json.loads(metadata_bytes)
    if original['global']['core:license']!='https://creativecommons.org/licenses/by/4.0/':
        raise RuntimeError('Source license changed; inspect before downloading')
    request=Request(BASE+NAME+'.sigmf-data',headers={'Range':'bytes=0-262143','User-Agent':'ETHERGLASS-research/0.1'})
    with urlopen(request,timeout=60) as response:
        if response.status!=206 or not response.headers.get('Content-Range','').startswith('bytes 0-262143/'):
            raise RuntimeError('Server did not honor the bounded byte range; refusing full download')
        data=response.read(262145)
        content_range=response.headers['Content-Range']
    if len(data)!=262144: raise RuntimeError('Unexpected partial-file size')
    name='camras-danuri-65536'
    provenance={'source_url':BASE+NAME+'.sigmf-data','source_metadata_url':BASE+NAME+'.sigmf-meta',
                'source_listing':BASE,'source_author':'Stichting CAMRAS',
                'license':'https://creativecommons.org/licenses/by/4.0/',
                'modifications':'First65536 complex samples, original ci16_le bytes, no scaling/filtering. Derived sidecar updated to SigMF1.2.6 and cropped dataset checksum. Original metadata retained separately.',
                'source_sample_interval':[0,65536],'content_range':content_range,
                'sha256':hashlib.sha256(data).hexdigest(),'clean_reference':False,
                'use':'External stability and abstention case only, no fitting or threshold changes. One recording session; adjacent windows correlated.'}
    meta={'global':{'core:datatype':'ci16_le','core:sample_rate':500000.,'core:version':'1.2.6',
                    'core:description':'Danuri observation excerpt, CAMRAS Dwingeloo radio telescope,2022-08-07. Cropped first65536samples.',
                    'core:author':'Stichting CAMRAS','core:license':provenance['license'],
                    'core:sha512':hashlib.sha512(data).hexdigest(),
                    'core:extensions':[{'name':'etherglass','version':'0.1.0','optional':True}],
                    'etherglass:provenance':provenance},
          'captures':[{'core:sample_start':0,'core:frequency':2260830000.,
                       'core:datetime':original['captures'][0]['core:datetime']}],'annotations':[]}
    (args.output/(name+'.sigmf-data')).write_bytes(data)
    (args.output/(name+'.sigmf-meta')).write_text(json.dumps(meta,indent=2)+'\n')
    (args.output/'camras-original-metadata.json').write_bytes(metadata_bytes)
    (args.output/'camras-danuri.manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__': main()
