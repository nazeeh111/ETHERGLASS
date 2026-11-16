"""Compact interference-aware frequency-domain networks.

This is a local, task-specific architecture, not an RF Challenge reproduction.
The U-Net comparator reimplements the generic encoder/decoder pattern.
"""
from __future__ import annotations
import torch
from torch import nn
from .spectral import FRAME_SIZE, FEATURE_CHANNELS, band_prior, spectral_features


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
