"""Long-context own Fourier vocoder refinement; lightweight training-only MPD.

MPD concept: Kong et al., HiFi-GAN, arXiv:2010.05646. Own reduced
implementation, not a faithful HiFi-GAN reproduction or a novelty claim.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mandia_voice import Discriminator, spectral_loss, to_mel

CORE = 96
CONTEXT = 96

def receptive_field(model):
    return 1 + sum((b.conv.kernel_size[0]-1)*b.conv.dilation[0] for b in model.blocks)

def eligible(sample, config):
    total=CORE+2*CONTEXT
    return len(sample['mel'])>=total+1 and len(sample['wave'])>=total*config.hop

def crop(samples, config, sampler, training=True):
    """Keep 96 mel frames on BOTH sides of the scored central waveform.

    Only real, unpadded centers contribute to losses; frame/sample offsets agree.
    """
    total=CORE+2*CONTEXT; mels=[];waves=[]
    for sample in samples:
        last=min(len(sample['mel'])-total-1,len(sample['wave'])//config.hop-total)
        if last<0: raise ValueError('Insufficient true context; no padded training crop')
        start=sampler.randint(0,last) if training else last//2
        mels.append(sample['mel'][start:start+total+1].float())
        center=(start+CONTEXT)*config.hop
        waves.append(sample['wave'][center:center+CORE*config.hop].float())
    return torch.stack(mels),torch.stack(waves)

def center_wave(model, mel):
    full=model(mel)
    begin=CONTEXT*model.config.hop
    return full[:,begin:begin+CORE*model.config.hop]

class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__();self.period=period
        channels=[1,8,32,64,128,1]
        self.layers=nn.ModuleList([nn.utils.parametrizations.weight_norm(nn.Conv2d(a,b,(5,1),stride=(3 if i<4 else 1,1),padding=(2,0))) for i,(a,b) in enumerate(zip(channels,channels[1:]))])
    def forward(self,wave):
        wave=wave[:,None];remaining=wave.shape[-1]%self.period
        if remaining:wave=F.pad(wave,(0,self.period-remaining),mode='reflect')
        h=wave.reshape(len(wave),1,-1,self.period);out=[]
        for i,layer in enumerate(self.layers):
            h=layer(h);out.append(h)
            if i<len(self.layers)-1:h=F.leaky_relu(h,.1)
        return out

class QualityDiscriminator(nn.Module):
    def __init__(self):
        super().__init__();self.spectral=Discriminator();self.periods=nn.ModuleList([PeriodDiscriminator(p) for p in [2,3,5,7,11]])
    def forward(self,wave):return self.spectral(wave)+[d(wave) for d in self.periods]

def reconstruction(fake,real,config):
    return spectral_loss(fake,real)+5*F.l1_loss(to_mel(fake,config),to_mel(real,config))

def generator_loss(fake,real,discriminator,config,adversarial=True):
    recon=reconstruction(fake,real,config)
    if not adversarial:return recon
    generated=discriminator(fake)
    with torch.no_grad():reference=discriminator(real)
    gan=sum((a[-1]-1).square().mean() for a in generated)/len(generated)
    matching=sum(sum(F.l1_loss(x,y) for x,y in zip(a[:-1],b[:-1]))/(len(a)-1) for a,b in zip(generated,reference))/len(generated)
    return recon+gan+2*matching

def discriminator_loss(fake,real,discriminator):
    generated=discriminator(fake.detach());reference=discriminator(real)
    return sum(a[-1].square().mean()+(b[-1]-1).square().mean() for a,b in zip(generated,reference))/len(generated)
