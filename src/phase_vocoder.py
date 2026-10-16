"""Own existing iSTFT generator, unchanged parameterization, consistency loss.

Short-time spectral consistency is published prior art (e.g. APNet,
Ai & Ling, arXiv:2305.07952). This normalized complex projection loss is an
experimental operational choice, not an APNet reproduction or echo detector.
"""
import torch
from mandia_voice import FourierVocoder

def project(spectrum,config,window):
    wave=torch.istft(spectrum,config.n_fft,config.hop,window=window,center=True)
    restored=torch.stft(wave,config.n_fft,config.hop,window=window,center=True,return_complex=True)
    return wave,restored

def relative_inconsistency(spectrum,restored,start=3,end=None):
    end=min(spectrum.shape[-1],restored.shape[-1])-3 if end is None else end
    a=spectrum[:,:,start:end];b=restored[:,:,start:end]
    # Normalize to avoid loud examples dominating. Keep the normalization fixed
    # for this gradient, and retain spectral/mel/GAN losses against zero collapse.
    return (a-b).abs().square().mean()/a.abs().square().mean().detach().clamp_min(1e-6)

class ConsistentVocoder(FourierVocoder):
    def spectrum(self,mel):
        x=self.input(mel);mask=torch.ones(x.shape[:2],dtype=torch.bool,device=mel.device)
        for block in self.blocks:x=block(x,mask)
        logmag,phase=self.output(x).chunk(2,-1);mag=logmag.clamp(-10,7).exp()
        return torch.complex(mag*phase.cos(),mag*phase.sin()).transpose(1,2)
    def forward(self,mel):
        return torch.istft(self.spectrum(mel),self.config.n_fft,self.config.hop,window=self.window,center=True)
    def waveform_and_consistency(self,mel,start,end):
        spec=self.spectrum(mel);wave,restored=project(spec,self.config,self.window)
        return wave,relative_inconsistency(spec,restored,start,end)
