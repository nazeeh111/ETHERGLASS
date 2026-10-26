import numpy as np
import pytest
from etherglass.synthetic import generate_example,rrc_taps
from etherglass.metrics import waveform_metrics,synthetic_symbol_metrics
from etherglass.separation import separate


def test_deterministic_exact_mixture_and_snr():
    a=generate_example(42,n=4096,snr_db=20,sir_db=-5)
    b=generate_example(42,n=4096,snr_db=20,sir_db=-5)
    np.testing.assert_array_equal(a["mixture"],b["mixture"])
    np.testing.assert_allclose(a["mixture"],a["target"]+a["interference"]+a["noise"],atol=2e-7)
    p=lambda x:np.mean(abs(x)**2)
    assert abs(10*np.log10(p(a["target"])/p(a["noise"]))-20)<1e-4
    assert abs(10*np.log10(p(a["target"])/p(a["interference"]))+5)<1e-4


def test_rrc_limits_and_spectral_occupancy():
    h=rrc_taps()
    assert np.isfinite(h).all()
    np.testing.assert_allclose(h,h[::-1],atol=1e-15)
    assert abs(sum(h*h)-1)<1e-12
    ex=generate_example(77,n=16384)
    p=abs(np.fft.fft(ex["target"]))**2
    assert p[abs(np.fft.fftfreq(len(p)))>.11].sum()/p.sum()<.01


def test_metric_alignment_and_zero_signal_failure():
    x=generate_example(13)["target"]
    m=waveform_metrics(x,x*(2+3j))
    assert m["si_sdr_db"]>120
    assert m["waveform_evm_pct"]<1e-4
    with pytest.raises(ValueError): waveform_metrics(np.zeros(10),np.zeros(10))
    assert waveform_metrics(x,np.zeros_like(x))["si_sdr_db"]==-120


def test_oracle_symbol_reference_metrics_and_no_input_mutation():
    ex=generate_example(40,n=4096,split="nominal")
    z=ex["target"].astype(np.complex128)
    before=z.copy()
    m=synthetic_symbol_metrics(z,ex["symbols"],ex["metadata"])
    np.testing.assert_array_equal(z,before)
    assert m["oracle_ser"]==0
    assert m["oracle_ber"]==0
    assert m["oracle_symbol_evm_pct"]<2


def test_separation_consistency_and_required_prior():
    ex=generate_example(901,n=3001)
    with pytest.raises(ValueError): separate(ex["mixture"],1e6)
    result=separate(ex["mixture"],1e6,symbol_rate=125000,method="wiener")
    np.testing.assert_allclose(sum(result["components"]),ex["mixture"],atol=5e-7)
    assert len(result["components"][0])==3001
    assert result["diagnostics"]["mixture_consistency_max_abs"]<5e-7
    short=separate(ex["mixture"][:128],1e6,symbol_rate=125000,method="wiener")
    assert short["diagnostics"]["short_record_unvalidated"]
    assert "Short record" in short["diagnostics"]["warning"]


def test_learned_checkpoint_and_nonoracle_real_input():
    from etherglass.separation import DEFAULT_CHECKPOINT
    if not DEFAULT_CHECKPOINT.is_file(): pytest.skip("checkpoint not present in source checkout")
    x=generate_example(92842,n=4096,split="test")["mixture"]
    r=separate(x,1e6,symbol_rate=125000)
    assert r["model"]["trained"]
    assert np.isfinite(r["components"][0]).all()
    assert not np.array_equal(x,r["components"][0])


def test_large_finite_iq_normalizes_without_power_overflow():
    from etherglass.models import spectral_features
    x=generate_example(135,n=2048)["mixture"]*np.float32(1e30)
    with np.errstate(over="raise",invalid="raise"):
        features,spectrum,scale=spectral_features(x)
        result=separate(x,1e6,symbol_rate=125000,method="notch")
    assert np.isfinite(features).all() and np.isfinite(spectrum).all()
    assert np.isfinite(scale)
    assert np.isfinite(result["diagnostics"]["target_power_fraction"])
    assert np.isfinite(result["components"][0]).all()


@pytest.mark.parametrize("dilation",[1,2,4,8])
def test_cpu_gemm_convolution_matches_native_output_and_gradients(dilation):
    import torch
    from etherglass.models import CPUConv1d
    torch.manual_seed(731)
    fast=CPUConv1d(3,5,5,padding=2*dilation,dilation=dilation)
    native=torch.nn.Conv1d(3,5,5,padding=2*dilation,dilation=dilation)
    native.load_state_dict(fast.state_dict())
    x1=torch.randn(2,3,128,requires_grad=True);x2=x1.detach().clone().requires_grad_(True)
    a=fast(x1);b=native(x2)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    a.square().mean().backward();b.square().mean().backward()
    torch.testing.assert_close(x1.grad,x2.grad,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(fast.weight.grad,native.weight.grad,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(fast.bias.grad,native.bias.grad,rtol=2e-5,atol=2e-6)
