"""Mandia Voix: experimental, random-init acoustic model and Fourier vocoder.

Original implementation of published ideas (MAS, conditional flow matching,
depthwise residual convolution and iSTFT); no claim of architectural novelty.
No pretrained weights or remote code are loaded by this module.
"""
from dataclasses import dataclass, asdict
import math
import unicodedata
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    speakers: int = 2
    vocab: int = 1026  # UTF-8 bytes + PAD and inter-byte blank; no unknown tokens
    dim: int = 256
    encoder_layers: int = 6
    flow_layers: int = 12
    expansion: int = 3
    kernel: int = 7
    mel: int = 80
    sample_rate: int = 24000
    n_fft: int = 1024
    hop: int = 256
    vocoder_dim: int = 128
    vocoder_layers: int = 8
    depthwise: bool = True
    encoder_attention: bool = True


def canonical_text(text):
    table = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-', '…': '...'})
    return ' '.join(unicodedata.normalize('NFC', text.translate(table)).casefold().split())


from phoneme_frontend import tokenize
from fast_alignment import maximum_path, beta_binomial_prior


def mask_for(lengths, maximum=None):
    maximum = int(lengths.max()) if maximum is None else maximum
    return torch.arange(maximum, device=lengths.device)[None, :] < lengths[:, None]


class Block(nn.Module):
    def __init__(self, dim, expansion=3, kernel=7, dilation=1, depthwise=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel, padding=(kernel//2)*dilation,
                              dilation=dilation, groups=dim if depthwise else 1)
        self.up = nn.Linear(dim, dim * expansion)
        self.down = nn.Linear(dim * expansion, dim)
        self.scale = nn.Parameter(torch.ones(dim) * 0.1)

    def forward(self, x, mask):
        x = x * mask[..., None]
        y = self.norm(x) * mask[..., None]
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        y = self.down(F.gelu(self.up(y))) * self.scale
        return (x + y) * mask[..., None]


@torch.no_grad()
def monotonic_path(scores):
    """Maximum-score surjective monotone path [frames, tokens], CPU DP.

    Every frame attends to one symbol; every symbol receives >=1 frame.
    The DP itself calls no recognizer. In this experiment the caller supplies soft guidance from automatic ASR word timestamps.
    """
    scores = scores.detach().float().cpu()
    frames, tokens = scores.shape
    if not 0 < tokens <= frames:
        raise ValueError('Audio must have at least as many frames as text symbols')
    neg = float('-inf')
    dp = torch.full((frames, tokens), neg)
    previous = torch.zeros((frames, tokens), dtype=torch.bool)
    dp[0, 0] = scores[0, 0]
    for t in range(1, frames):
        stay = dp[t-1]
        advance = F.pad(stay[:-1], (1, 0), value=neg)
        take = advance > stay
        dp[t] = torch.maximum(stay, advance) + scores[t]
        dp[t, min(t+1, tokens):] = neg
        previous[t] = take
    if not torch.isfinite(dp[-1, -1]):
        raise ValueError('No complete monotonic alignment')
    indices = torch.zeros(frames, dtype=torch.long)
    j = tokens - 1
    for t in range(frames - 1, -1, -1):
        indices[t] = j
        if previous[t, j]:
            j -= 1
    return indices


class Acoustic(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        c = config
        self.embedding = nn.Embedding(c.vocab, c.dim, padding_idx=0)
        self.speaker = nn.Embedding(c.speakers, c.dim)
        self.encoder = nn.ModuleList([Block(c.dim, c.expansion, c.kernel,
            2**(i % 4), c.depthwise) for i in range(c.encoder_layers)])
        self.attention = nn.ModuleList([nn.MultiheadAttention(c.dim, 4, batch_first=True)
            for _ in range(c.encoder_layers//2)]) if c.encoder_attention else nn.ModuleList()
        self.attention_norm = nn.ModuleList([nn.LayerNorm(c.dim) for _ in self.attention])
        self.mean = nn.Linear(c.dim, c.mel)
        self.duration = nn.Sequential(nn.Linear(c.dim, c.dim), nn.GELU(), nn.Linear(c.dim, 1))
        self.flow_in = nn.Linear(c.mel * 2, c.dim)
        self.time = nn.Sequential(nn.Linear(32, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim))
        self.flow = nn.ModuleList([Block(c.dim, c.expansion, c.kernel,
            2**(i % 4), c.depthwise) for i in range(c.flow_layers)])
        self.flow_out = nn.Linear(c.dim, c.mel)

    def speaker_vector(self, batch, device, speaker_ids=None):
        if speaker_ids is None:
            if self.config.speakers > 1:
                raise ValueError('speaker_ids are required for the conditional model')
            speaker_ids = torch.zeros(batch, dtype=torch.long, device=device)
        if speaker_ids.shape != (batch,) or speaker_ids.dtype != torch.long:
            raise ValueError('Expected one integer speaker index per item')
        if int(speaker_ids.min()) < 0 or int(speaker_ids.max()) >= self.config.speakers:
            raise ValueError('Unknown speaker index')
        return self.speaker(speaker_ids)

    def encode(self, text, lengths, speaker_ids=None):
        mask = mask_for(lengths, text.shape[1])
        x = self.embedding(text) + self.speaker_vector(len(text), text.device, speaker_ids)[:, None]
        # Deterministic position information, no additional learned parameters.
        position = torch.arange(text.shape[1], device=text.device)[:, None]
        frequency = torch.exp(torch.arange(0,self.config.dim,2,device=text.device)*(-math.log(10000)/self.config.dim))
        pe = torch.zeros(text.shape[1],self.config.dim,device=text.device)
        pe[:,0::2] = (position*frequency).sin(); pe[:,1::2] = (position*frequency).cos()
        x = (x + .1*pe[None])*mask[...,None]
        for i, block in enumerate(self.encoder):
            x = block(x, mask)
            if i%2==1 and self.config.encoder_attention:
                h = self.attention_norm[i//2](x)
                h = self.attention[i//2](h, h, h, key_padding_mask=~mask, need_weights=False)[0]
                x = (x+h) * mask[..., None]
        return self.mean(x) * mask[..., None], self.duration(x.detach()).squeeze(-1), mask

    def velocity(self, x, mean, time, mask, speaker_ids=None):
        freqs = torch.exp(torch.linspace(0, math.log(1000), 16, device=x.device))
        phase = time[:, None] * freqs[None]
        embedding = self.time(torch.cat([phase.sin(), phase.cos()], -1))
        h = (self.flow_in(torch.cat([x, mean], -1)) + embedding[:, None] + self.speaker_vector(len(x),x.device,speaker_ids)[:,None]) * mask[..., None]
        for block in self.flow:
            h = block(h, mask)
        return self.flow_out(h) * mask[..., None]

    def losses(self, text, text_lengths, mel, mel_lengths, generator=None, speaker_ids=None, anchors=None):
        means, log_duration, text_mask = self.encode(text, text_lengths, speaker_ids)
        aligned = torch.zeros_like(mel)
        durations = torch.zeros_like(log_duration)
        for b in range(text.shape[0]):
            nt, nf = int(text_lengths[b]), int(mel_lengths[b])
            # Gaussian acoustic likelihood plus a decaying positional warmup prior.
            with torch.no_grad():
                x=mel[b,:nf].float();m=means[b,:nt].float()
                scores=-.5*(x.square().sum(-1,keepdim=True)+m.square().sum(-1)[None]-2*x@m.T)
                weight=max(0.,1.-getattr(self,'training_step',0)/6000.)
                if weight:
                    scores=scores+weight*torch.as_tensor(beta_binomial_prior(nf,nt).copy(),device=mel.device)
                path=torch.from_numpy(maximum_path(scores.detach().cpu().numpy())).to(text.device)
            aligned[b, :nf] = means[b, path]
            durations[b, :nt] = torch.bincount(path, minlength=nt)
        mask = mask_for(mel_lengths, mel.shape[1])
        denom = mask.sum() * mel.shape[-1]
        prior = ((mel - aligned).square() * mask[..., None]).sum() / denom
        duration = ((log_duration - torch.log(durations.clamp_min(1))).square()
                    * text_mask).sum() / text_mask.sum()
        predicted_frames = (log_duration.clamp(-3, math.log(100)).exp()*text_mask).sum(-1)
        duration_total = (predicted_frames.clamp_min(1).log()-mel_lengths.float().log()).square().mean()
        # Flow learns fixed 192-frame acoustic windows; full utterances supervise the encoder.
        window=min(192,mel.shape[1]);flow_mel=mel.new_zeros((len(mel),window,mel.shape[-1]));flow_mean=torch.zeros_like(flow_mel)
        flow_lengths=mel_lengths.clamp(max=window)
        for b in range(len(mel)):
            available=max(0,int(mel_lengths[b])-window)
            offset=int(torch.randint(available+1,(),device=mel.device,generator=generator)) if available else 0
            length=int(flow_lengths[b]);flow_mel[b,:length]=mel[b,offset:offset+length];flow_mean[b,:length]=aligned[b,offset:offset+length]
        mask=mask_for(flow_lengths,window);denom=mask.sum()*mel.shape[-1]
        noise = torch.randn(flow_mel.shape, device=mel.device, generator=generator)
        time = torch.rand((mel.shape[0],), device=mel.device, generator=generator)
        sigma = 1e-4
        xt = (1 - (1-sigma)*time[:, None, None]) * noise + time[:, None, None] * flow_mel
        target = flow_mel - (1-sigma)*noise
        velocity = self.velocity(xt, flow_mean, time, mask, speaker_ids)
        flow = ((velocity - target).square() * mask[..., None]).sum() / denom
        return {'prior': prior, 'duration': duration, 'duration_total': .5*duration_total, 'flow': flow}

    @torch.no_grad()
    def synthesize(self, text, text_lengths, steps=8, speed=1.0, seed=42, max_frames=5000, speaker_ids=None):
        if steps < 1 or speed <= 0:
            raise ValueError('steps and speed must be positive')
        means, log_duration, mask = self.encode(text, text_lengths, speaker_ids)
        durations = torch.ceil(log_duration.clamp(0, math.log(100)).exp()/speed).long() * mask
        lengths = durations.sum(-1)
        if int(lengths.max()) > max_frames:
            raise ValueError('Output exceeds frame budget; segment the text instead of truncating')
        aligned = means.new_zeros((text.shape[0], int(lengths.max()), self.config.mel))
        for b in range(text.shape[0]):
            expanded = torch.repeat_interleave(means[b], durations[b], dim=0)
            aligned[b, :len(expanded)] = expanded
        frame_mask = mask_for(lengths)
        rng = torch.Generator(device=text.device).manual_seed(seed)
        x = torch.randn(aligned.shape, device=text.device, generator=rng)
        x *= frame_mask[..., None]
        for i in range(steps):
            time = x.new_full((len(text),), i/steps)
            x = (x + self.velocity(x, aligned, time, frame_mask, speaker_ids)/steps) * frame_mask[..., None]
        return x, lengths


class FourierVocoder(nn.Module):
    """Full-frequency iSTFT head, inspired by Vocos; not an iSTFTNet reproduction."""
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        c = config
        self.input = nn.Linear(c.mel, c.vocoder_dim)
        self.blocks = nn.ModuleList([Block(c.vocoder_dim, 3, 7,
            2**(i % 4), c.depthwise) for i in range(c.vocoder_layers)])
        self.output = nn.Linear(c.vocoder_dim, (c.n_fft//2+1)*2)
        self.register_buffer('window', torch.hann_window(c.n_fft))

    def forward(self, mel):
        x = self.input(mel)
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        logmag, phase = self.output(x).chunk(2, dim=-1)
        magnitude = logmag.clamp(-10, 7).exp()
        spectrum = torch.complex(magnitude*phase.cos(), magnitude*phase.sin()).transpose(1, 2)
        return torch.istft(spectrum, self.config.n_fft, self.config.hop,
                           window=self.window, center=True)


class Discriminator(nn.Module):
    """Two STFT resolutions; research control, not the original Vocos adversary."""
    def __init__(self):
        super().__init__()
        self.n_ffts = (512, 1024)
        self.nets = nn.ModuleList([nn.ModuleList([
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Conv2d(16, 32, 3, stride=(2, 2), padding=1),
            nn.Conv2d(32, 64, 3, stride=(2, 2), padding=1),
            nn.Conv2d(64, 1, 3, padding=1)]) for _ in self.n_ffts])

    def forward(self, wave):
        result = []
        for fft, net in zip(self.n_ffts, self.nets):
            window = torch.hann_window(fft, device=wave.device)
            h = torch.stft(wave, fft, fft//4, window=window, return_complex=True)
            h = h.abs().clamp_min(1e-5).log().unsqueeze(1)
            features = []
            for layer in net:
                h = layer(h)
                features.append(h)
                h = F.leaky_relu(h, 0.2)
            result.append(features)
        return result


def mel_filter(config):
    # HTK mel, area normalization, explicit fixed definition in the run.
    c = config
    melmax = 2595 * math.log10(1 + (c.sample_rate/2)/700)
    edges = 700 * (10**(torch.linspace(0, melmax, c.mel+2)/2595) - 1)
    hz = torch.linspace(0, c.sample_rate/2, c.n_fft//2+1)
    left = (hz[None]-edges[:-2,None])/(edges[1:-1]-edges[:-2])[:,None]
    right = (edges[2:,None]-hz[None])/(edges[2:]-edges[1:-1])[:,None]
    return torch.minimum(left, right).clamp_min(0) * (2/(edges[2:]-edges[:-2]))[:,None]


def to_mel(wave, config=Config()):
    window = torch.hann_window(config.n_fft, device=wave.device)
    spec = torch.stft(wave, config.n_fft, config.hop, window=window,
                      center=True, return_complex=True).abs()
    bank = mel_filter(config).to(wave.device)
    return (bank @ spec).clamp_min(1e-5).log().transpose(-1, -2)


def spectral_loss(fake, real):
    loss = fake.new_zeros(())
    for fft in (256, 512, 1024):
        window = torch.hann_window(fft, device=fake.device)
        x = torch.stft(fake, fft, fft//4, window=window, return_complex=True).abs().clamp_min(1e-5)
        y = torch.stft(real, fft, fft//4, window=window, return_complex=True).abs().clamp_min(1e-5)
        loss = loss + F.l1_loss(x.log(), y.log()) + (x-y).norm()/y.norm().clamp_min(1e-5)
    return loss/3


def parameter_counts(config=Config()):
    return {name: sum(p.numel() for p in model.parameters()) for name, model in
            [('acoustic', Acoustic(config)), ('vocoder', FourierVocoder(config)),
             ('training_discriminator', Discriminator())]}
