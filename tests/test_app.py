import io
import json
import zipfile
import numpy as np
from fastapi.testclient import TestClient
from etherglass.app import app
from etherglass.io import encode_sigmf,load_bytes

client=TestClient(app)


def test_actual_input_import_analysis_and_export():
    fs=1e6
    x=np.exp(2j*np.pi*.125*np.arange(4096)).astype('complex64')
    raw,meta=encode_sigmf(x,fs,915e6)
    r=client.post('/api/import',files={'data_file':('tone.sigmf-data',raw),
        'meta_file':('tone.sigmf-meta',json.dumps(meta).encode())})
    assert r.status_code==200,r.text
    a=r.json()
    s=a['spectrum']; peak=s['frequency_hz'][int(np.argmax(s['power_db']))]
    assert peak==125000
    assert a['classification']['status']=='unknown'
    ident=a['capture_id']
    a=client.post('/api/analyze',json={'capture_id':ident,'start_sample':512,'stop_sample':2048,'inspect_only':True})
    assert a.status_code==200,a.text
    assert a.json()['num_samples']==1536
    z=client.get(f'/api/export/{ident}?component=mixture')
    assert z.status_code==200
    with zipfile.ZipFile(io.BytesIO(z.content)) as f:
        cap=load_bytes(f.read('capture.sigmf-data'),metadata=json.loads(f.read('capture.sigmf-meta')))
        np.testing.assert_array_equal(cap.samples,x[512:2048])
    assert client.get(f'/api/export/{ident}?component=target').status_code==409
    assert client.get(f'/api/export/{ident}?component=foo').status_code==422


def test_demo_and_reconstruction_do_not_use_truth_at_inference():
    a=client.get('/api/demo')
    assert a.status_code==200,a.text
    data=a.json()
    r=client.post('/api/analyze',json={'capture_id':data['capture_id'],
        'symbol_rate':125000,'method':'matched','start_sample':0,'stop_sample':4096})
    assert r.status_code==200,r.text
    b=r.json()
    assert len(b['separation']['components'])==2
    assert 'reference_metrics' in b['separation']
    assert b['separation']['components'][0]['waveform']['i']!=b['waveform']['i']


def test_error_responses_and_loopback_host_guard():
    assert client.post('/api/import',files={'data_file':('x',b'123')}).status_code==422
    assert client.post('/api/analyze',json={'capture_id':'unknown'}).status_code==404
    assert client.post('/api/analyze',content='{"capture_id":"x","target_center":Infinity}',headers={'Content-Type':'application/json'}).status_code==422
    assert client.get('/',headers={'host':'untrusted.example'}).status_code==400
    assert client.get('/').status_code==200


def test_nonfinite_metadata_rejected_and_high_dynamic_range_measured():
    x=(1e30*np.exp(2j*np.pi*.125*np.arange(2048))).astype('complex64')
    raw,meta=encode_sigmf(x,1e6,0)
    response=client.post('/api/import',files={'data_file':('x',raw),
        'meta_file':('x.json',json.dumps(meta).encode())})
    assert response.status_code==200,response.text
    assert np.isfinite(response.json()['spectrum']['power_db']).all()
    meta['unrecognized_field']={'nested':float('nan')}
    response=client.post('/api/import',files={'data_file':('x',raw),
        'meta_file':('x.json',json.dumps(meta).encode())})
    assert response.status_code==422


def test_unknown_required_extension_refused_but_optional_retained():
    raw,meta=encode_sigmf(np.ones(2048,dtype='complex64'),1e6,0)
    meta['global']['core:extensions']=[{'name':'unsupported','version':'1.0.0','optional':False}]
    for optional,expected in [(False,422),(True,200)]:
        meta['global']['core:extensions'][0]['optional']=optional
        r=client.post('/api/import',files={'data_file':('x',raw),
             'meta_file':('x.json',json.dumps(meta).encode())})
        assert r.status_code==expected,r.text
