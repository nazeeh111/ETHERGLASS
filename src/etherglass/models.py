"""Compact interference-aware frequency-domain networks.

This is a local, task-specific architecture, not an RF Challenge reproduction.
The U-Net comparator reimplements the generic encoder/decoder pattern.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d
import torch
from torch import nn

FRAME_SIZE=2048
FEATURE_CHANNELS=7


class CPUConv1d(nn.Conv1d):
    """The same convolution via GEMM on CPU, avoiding slow dilated kernels.

    Weights/state keys and convolution mathematics are unchanged. The native
    PyTorch path remains for other devices or unsupported convolution options.
    Floating-point reduction order can differ slightly between the two paths.
    """
    use_gemm=True
    def forward(self,x):
        if (not self.use_gemm or x.device.type!="cpu" or self.groups!=1
                or self.stride!=(1,) or self.padding_mode!="zeros"
                or not isinstance(self.padding,tuple)):
            return super().forward(x)
        padding=self.padding[0];dilation=self.dilation[0];kernel=self.kernel_size[0]
        windows=torch.nn.functional.pad(x,(padding,padding)).unfold(-1,dilation*(kernel-1)+1,1)[...,::dilation]
        windows=windows.permute(0,2,1,3).flatten(2)
        return torch.nn.functional.linear(windows,self.weight.flatten(1),self.bias).transpose(1,2)


def band_prior(n: int, sps: float = 8.) -> np.ndarray:
    f=np.abs(np.fft.fftshift(np.fft.fftfreq(n)))
    # Full gain in the RRC support; smooth numerical margin outside its edge.
    edge=.675/sps
    return (.5*(1+np.cos(np.pi*np.clip((f-edge)/(.12/sps),0,1)))).astype(np.float32)


def spectral_features(x: np.ndarray, sps: float=8.) -> tuple[np.ndarray,np.ndarray,float]:
    """Features use only mixture IQ and explicit symbol-rate prior, never truth."""
    x=np.asarray(x,dtype=np.complex64)
    n=len(x)
    # Accumulate in float64: finite cf32 captures can have amplitudes whose
    # squared magnitude overflows float32 before normalization.
    scale=float(np.sqrt(np.mean(np.abs(x.astype(np.complex128))**2))+1e-8)
    spec=np.fft.fftshift(np.fft.fft(x/scale,norm="ortho"))
    power=np.abs(spec)**2
    log=np.log1p(power).astype(np.float32)
    floor=np.maximum(uniform_filter1d(power,size=127,mode="wrap"),1e-6)
    whiten=spec/np.sqrt(floor)
    contrast7=log-uniform_filter1d(log,size=7,mode="wrap")
    contrast31=log-uniform_filter1d(log,size=31,mode="wrap")
    contrast127=log-uniform_filter1d(log,size=127,mode="wrap")
    feat=np.stack([np.clip(whiten.real,-8,8),np.clip(whiten.imag,-8,8),log,
                   contrast7,contrast31,contrast127,band_prior(n,sps)]).astype(np.float32)
    return feat,spec.astype(np.complex64),scale


class ResidualBlock(nn.Module):
    def __init__(self, width:int,dilation:int):
        super().__init__()
        self.net=nn.Sequential(CPUConv1d(width,width,5,padding=2*dilation,dilation=dilation),
                               nn.SiLU(),CPUConv1d(width,width,1),nn.SiLU())
    def forward(self,x): return x+self.net(x)


class InterferenceNet(nn.Module):
    """7-channel learned IQ/contrast representation + analytic support constraint."""
    def __init__(self,width:int=16,no_dsp:bool=False):
        super().__init__()
        self.no_dsp=no_dsp
        self.input=nn.Sequential(CPUConv1d(FEATURE_CHANNELS,width,5,padding=2),nn.SiLU())
        self.blocks=nn.Sequential(*(ResidualBlock(width,d) for d in [1,2,4,8]))
        self.out=CPUConv1d(width,1,1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias,2.)
    def forward(self,x):
        if self.no_dsp:
            x=torch.cat([x[:,:3],torch.zeros_like(x[:,3:6]),x[:,6:]],dim=1)
        gain=torch.sigmoid(self.out(self.blocks(self.input(x))))[:,0]
        return gain*x[:,6]


class ConvPair(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.net=nn.Sequential(CPUConv1d(inp,out,5,padding=2),nn.SiLU(),
                               CPUConv1d(out,out,5,padding=2),nn.SiLU())
    def forward(self,x): return self.net(x)


class SpectralUNet(nn.Module):
    """Small U-Net baseline, same input features and support prior as proposed net."""
    def __init__(self):
        super().__init__()
        self.enc1=ConvPair(7,12)
        self.enc2=ConvPair(12,24)
        self.bottleneck=ConvPair(24,32)
        self.dec2=ConvPair(56,24)
        self.dec1=ConvPair(36,12)
        self.out=CPUConv1d(12,1,1)
        self.pool=nn.AvgPool1d(2)
    def forward(self,x):
        a=self.enc1(x); b=self.enc2(self.pool(a)); c=self.bottleneck(self.pool(b))
        c=torch.nn.functional.interpolate(c,size=b.shape[-1],mode="linear",align_corners=False)
        d=self.dec2(torch.cat([c,b],1))
        d=torch.nn.functional.interpolate(d,size=a.shape[-1],mode="linear",align_corners=False)
        gain=torch.sigmoid(self.out(self.dec1(torch.cat([d,a],1))))[:,0]
        return gain*x[:,6]


def make_model(kind:str="hybrid") -> nn.Module:
    if kind in {"hybrid","pretrained","scratch"}: return InterferenceNet()
    if kind == "no_dsp": return InterferenceNet(no_dsp=True)
    if kind == "unet": return SpectralUNet()
    raise ValueError(f"unknown architecture: {kind}")
