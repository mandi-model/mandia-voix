"""Own controlled width study. Published ideas: Net2Wider, HiFi-GAN MSD.
No claim of novelty or exact reproduction. Added adversaries are training-only.
"""
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import replace
from phase_vocoder import ConsistentVocoder
from context_vocoder import QualityDiscriminator,reconstruction

@torch.no_grad()
def widen(base,seed=67):
    """Duplicate channels, preserve LayerNorm statistics and the initial function.

    Opposite perturbations of duplicate input columns cancel at initialization,
    while allowing gradients to break the channel symmetry on later updates.
    """
    config=replace(base.config,vocoder_dim=base.config.vocoder_dim*2)
    model=ConsistentVocoder(config).to(next(base.parameters()).device)
    old=base.state_dict();new=model.state_dict();rng=torch.Generator(device='cpu').manual_seed(seed)
    for key,value in old.items():
        if key=='input.weight' or key=='input.bias':new[key].copy_(value.repeat(2,*([1]*(value.ndim-1))))
        elif key=='output.weight':
            expanded=value.repeat(1,2)/2
            perturb=torch.randn(value.shape,generator=rng).to(value.device)*.001
            expanded[:,:value.shape[1]]+=perturb;expanded[:,value.shape[1]:]-=perturb;new[key].copy_(expanded)
        elif key=='output.bias' or key=='window':new[key].copy_(value)
        elif key.endswith('up.weight') or key.endswith('down.weight'):
            expanded=value.repeat(2,2)/2;perturb=torch.randn((value.shape[0]*2,value.shape[1]),generator=rng).to(value.device)*.001
            expanded[:,:value.shape[1]]+=perturb;expanded[:,value.shape[1]:]-=perturb;new[key].copy_(expanded)
        elif key.startswith('blocks.'):
            new[key].copy_(value.repeat(2,*([1]*(value.ndim-1))))
        else:raise ValueError('Unmapped parameter '+key)
    model.load_state_dict(new);return model

class WaveScale(nn.Module):
    def __init__(self):
        super().__init__();spec=[(1,16,15,1,1),(16,64,21,4,4),(64,128,21,4,8),(128,128,21,4,8),(128,128,5,1,1),(128,1,3,1,1)]
        self.layers=nn.ModuleList([nn.utils.parametrizations.weight_norm(nn.Conv1d(a,b,k,stride=s,padding=k//2,groups=g)) for a,b,k,s,g in spec])
    def forward(self,wave):
        x=wave[:,None];result=[]
        for i,layer in enumerate(self.layers):
            x=layer(x);result.append(x)
            if i<len(self.layers)-1:x=F.leaky_relu(x,.1)
        return result

class NaturalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__();self.previous=QualityDiscriminator();self.scales=nn.ModuleList([WaveScale() for _ in range(3)])
    def forward(self,wave):
        result=self.previous(wave);current=wave
        for i,scale in enumerate(self.scales):
            if i:current=F.avg_pool1d(current[:,None],4,2,padding=1)[:,0]
            result.append(scale(current))
        return result

def faithful_reconstruction(fake,real,config):
    old=reconstruction(fake,real,config);window=torch.hann_window(2048,device=fake.device)
    a=torch.stft(fake,2048,512,window=window,return_complex=True).abs().clamp_min(1e-5)
    b=torch.stft(real,2048,512,window=window,return_complex=True).abs().clamp_min(1e-5)
    return old+.5*(F.l1_loss(a.log(),b.log())+(a-b).norm()/b.norm().clamp_min(1e-5))

def natural_generator_loss(fake,real,disc,config,consistency,adversarial=True):
    loss=faithful_reconstruction(fake,real,config)+.25*consistency
    if not adversarial:return loss
    a=disc(fake)
    with torch.no_grad():b=disc(real)
    gan=sum((x[-1]-1).square().mean() for x in a)/len(a)
    fm=sum(sum(F.l1_loss(x,y) for x,y in zip(xs[:-1],ys[:-1]))/(len(xs)-1) for xs,ys in zip(a,b))/len(a)
    return loss+gan+2*fm
