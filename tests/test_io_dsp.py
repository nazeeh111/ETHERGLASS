import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from etherglass.io import Capture, decode_iq, encode_sigmf, export_zip, iter_iq_blocks, load_bytes, load_capture
from etherglass.dsp import spectrum, waterfall


def test_fft_axis_and_absolute_psd_power():
    fs, n, freq = 16384, 4096, 1024
    x = np.exp(2j*np.pi*freq*np.arange(n)/fs).astype(np.complex64)
    s = spectrum(x, fs)
    f, p = np.asarray(s['frequency_hz']), 10**(np.asarray(s['power_db'])/10)
    assert f[np.argmax(p)] == pytest.approx(freq)
    assert p.sum()*(f[1]-f[0]) == pytest.approx(1, rel=1e-5)
    w = waterfall(x, fs)
    assert len(w['power_db']) <= 128
    assert len(w['power_db'][0]) == len(w['frequency_hz'])
    assert w['frequency_hz'][np.argmax(w['power_db'][0])] == pytest.approx(freq)


@pytest.mark.parametrize('dtype,values,expected', [
    ('ci8', np.array([-128,127]*64,dtype='i1').tobytes(), -1+127j/128),
    ('cu8', np.array([0,255]*64,dtype='u1').tobytes(), -1+127j/128),
    ('ci16_be',np.array([-32768,16384]*64,dtype='>i2').tobytes(),-1+.5j),
    ('cf32_be',np.array([.25,-.75]*64,dtype='>f4').tobytes(),.25-.75j),
])
def test_complex_endianness_and_normalization(dtype,values,expected):
    x=decode_iq(values,dtype)
    assert np.allclose(x,expected)


def test_sigmf_round_trip_and_official_schema(tmp_path):
    from sigmf import SigMFFile
    x=np.exp(.03j*np.arange(2048)).astype(np.complex64)
    raw,meta=encode_sigmf(x,1e6,915e6,license='https://creativecommons.org/publicdomain/zero/1.0/')
    SigMFFile(metadata=meta).validate()
    cap=load_bytes(raw,metadata=meta)
    np.testing.assert_array_equal(x,cap.samples)
    assert cap.source_sha256==hashlib.sha256(raw).hexdigest()
    f=tmp_path/'test.sigmf-data'; f.write_bytes(raw)
    f.with_suffix('.sigmf-meta').write_text(json.dumps(meta))
    np.testing.assert_array_equal(load_capture(f).samples,x)
    np.testing.assert_array_equal(np.concatenate(list(iter_iq_blocks(f,'cf32_le',block_samples=177))),x)
    z=zipfile.ZipFile(io.BytesIO(export_zip(x,1e6,0,{'source_sha256':cap.source_sha256})))
    assert set(z.namelist())=={'capture.sigmf-data','capture.sigmf-meta','analysis.json'}
    SigMFFile(metadata=json.loads(z.read('capture.sigmf-meta'))).validate()


@pytest.mark.parametrize('kind',['truncated','nan','rate','missing','checksum','bounds','channels'])
def test_invalid_input_never_silently_fixes_metadata(kind):
    raw,meta=encode_sigmf(np.ones(128,dtype='complex64'),1e6,0)
    if kind=='truncated': raw=raw[:-1]
    elif kind=='nan': raw=np.array([float('nan')]*256,dtype='<f4').tobytes()
    elif kind=='rate': meta['global']['core:sample_rate']=0
    elif kind=='missing': del meta['global']['core:sample_rate']
    elif kind=='checksum': meta['global']['core:sha512']='0'*128
    elif kind=='bounds': meta['captures'][0]['core:sample_start']=128
    elif kind=='channels': meta['global']['core:num_channels']=2
    with pytest.raises(ValueError): load_bytes(raw,metadata=meta)


def test_capture_retuning_boundaries():
    from etherglass.analysis import analyze
    cap=Capture(np.ones(2048,dtype='complex64'),1e6,900e6,
        metadata={'captures':[{'core:sample_start':0,'core:frequency':900e6},
                              {'core:sample_start':1024,'core:frequency':901e6}]})
    with pytest.raises(ValueError,match='tuning boundary'): analyze(cap)
    result=analyze(cap,start_sample=1024,stop_sample=2048)
    assert result['center_frequency']==901e6


def test_center_frequency_is_explicit_even_when_sigmf_omits_optional_frequency():
    raw,meta=encode_sigmf(np.ones(128,dtype='complex64'),1e6,0)
    del meta['captures'][0]['core:frequency']
    with pytest.raises(ValueError,match='center frequency'):
        load_bytes(raw,metadata=meta)
    assert load_bytes(raw,metadata=meta,center_frequency=0).center_frequency==0


def test_derived_sigmf_pair_keeps_attribution_without_requiring_report_file():
    source={'metadata':{'global':{'core:author':'Original recording author',
                                  'core:description':'Public test source'}},
            'license':'https://creativecommons.org/licenses/by/4.0/'}
    blob=export_zip(np.ones(128,dtype='complex64'),1e6,0,
                    {'provenance':source,'exported_component':'target','source_sha256':'abc'},
                    license=source['license'])
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        meta=json.loads(z.read('capture.sigmf-meta'))
    assert meta['global']['core:author']=='Original recording author'
    assert meta['global']['core:license']==source['license']
    assert meta['global']['etherglass:provenance']['source']==source
    assert 'derived target' in meta['global']['core:description']


@pytest.mark.parametrize('sample_rate',[1e-308,1e308])
def test_extreme_sample_rates_rejected_before_analysis_overflows(sample_rate):
    with pytest.raises(ValueError,match='supported numerical range'):
        Capture(np.ones(2048,dtype='complex64'),sample_rate,0)
