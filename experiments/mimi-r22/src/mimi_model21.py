"""Small text-conditioned temporal/depth codec predictor, inspired by Moshi/CSM.
Own pilot implementation, not a reproduction or novelty claim.
Frozen lexical stem copied from our N; pretrained Mimi codebooks initialize inputs.
"""
import math,copy
import torch
from torch import nn
from torch.nn import functional as F
from mandia_voice import Block,Config,mask_for

class TextStem(nn.Module):
 def __init__(self,config):
  super().__init__();self.config=Config(**config);c=self.config
  self.embedding=nn.Embedding(c.vocab,c.dim,padding_idx=0);self.speaker=nn.Embedding(c.speakers,c.dim)
  self.encoder=nn.ModuleList([Block(c.dim,c.expansion,c.kernel,2**(i%4),c.depthwise) for i in range(c.encoder_layers)])
  self.attention=nn.ModuleList([nn.MultiheadAttention(c.dim,4,batch_first=True) for _ in range(c.encoder_layers//2)])
  self.attention_norm=nn.ModuleList([nn.LayerNorm(c.dim) for _ in self.attention]);self.mean=nn.Linear(c.dim,c.mel)
 def forward(self,ids,lengths):
  mask=mask_for(lengths,ids.shape[1]);x=self.embedding(ids)+self.speaker.weight[0][None,None]
  pos=torch.arange(ids.shape[1],device=ids.device)[:,None];freq=torch.exp(torch.arange(0,self.config.dim,2,device=ids.device)*(-math.log(10000)/self.config.dim));pe=torch.zeros(ids.shape[1],self.config.dim,device=ids.device);pe[:,0::2]=(pos*freq).sin();pe[:,1::2]=(pos*freq).cos();x=(x+.1*pe[None])*mask[...,None]
  for i,b in enumerate(self.encoder):
   x=b(x,mask)
   if i%2==1:
    h=self.attention_norm[i//2](x);h=self.attention[i//2](h,h,h,key_padding_mask=~mask,need_weights=False)[0];x=(x+h)*mask[...,None]
  return x,self.mean(x)*mask[...,None]

def coarse_alignment(mu,durations,frames):
 """Duration-weighted phonetic means in codec frame intervals; no speech trimming."""
 edges=F.pad(durations.float().cumsum(0),(1,0));width=edges[-1]/frames
 left=torch.arange(frames,device=mu.device).float()*width;right=left+width
 weights=(torch.minimum(right[:,None],edges[1:][None])-torch.maximum(left[:,None],edges[:-1][None])).clamp_min(0)
 return weights@mu/weights.sum(-1,keepdim=True).clamp_min(1e-6)

class MimiText21(nn.Module):
 def __init__(self,stem_config,dim=320,layers=6,books=8):
  super().__init__();self.settings=dict(stem_config=stem_config,dim=dim,layers=layers,books=books);self.books=books;self.dim=dim
  self.stem=TextStem(stem_config).requires_grad_(False).eval()
  self.duration=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,1));self.duration_domain=nn.Embedding(2,1);nn.init.zeros_(self.duration_domain.weight)
  self.code_embeddings=nn.ModuleList([nn.Embedding(2049,256) for _ in range(books)])
  self.audio_input=nn.Linear(256,dim);self.aligned_input=nn.Linear(80,dim);self.memory_input=nn.Linear(256,dim);self.domain=nn.Embedding(2,dim)
  self.temporal=nn.ModuleList([nn.TransformerDecoderLayer(dim,5,dim*3,dropout=.1,activation='gelu',batch_first=True,norm_first=True) for _ in range(layers)])
  self.final_norm=nn.LayerNorm(dim);self.depth_start=nn.Parameter(torch.zeros(256));self.depth_init=nn.Linear(dim,dim);self.depth=nn.GRUCell(256,dim)
  self.code_queries=nn.ModuleList([nn.Linear(dim,256) for _ in range(books)]);self.code_bias=nn.Parameter(torch.zeros(books,2048))
 def train(self,mode=True):super().train(mode);self.stem.eval();return self
 def log_durations(self,memory,domain):return self.duration(memory).squeeze(-1)+self.duration_domain(domain)
 def context(self,codes,memory,aligned,domains,audio_lengths,text_lengths):
  b,t,k=codes.shape;prev=F.pad(codes[:,:-1],(0,0,1,0),value=2048)
  x=sum(emb(prev[:,:,i]) for i,emb in enumerate(self.code_embeddings))/math.sqrt(self.books)
  if self.training:x=x*(torch.rand(b,t,1,device=x.device)>=.1).to(x.dtype)/.9
  h=self.audio_input(x)+self.aligned_input(aligned)+self.domain(domains)[:,None]
  p=torch.arange(t,device=h.device)[:,None];f=torch.exp(torch.arange(0,self.dim,2,device=h.device)*(-math.log(10000)/self.dim));pe=torch.zeros(t,self.dim,device=h.device);pe[:,0::2]=(p*f).sin();pe[:,1::2]=(p*f).cos();h=h+.1*pe[None]
  mem=self.memory_input(memory);causal=torch.ones(t,t,device=h.device,dtype=torch.bool).triu(1);apad=~mask_for(audio_lengths,t);tpad=~mask_for(text_lengths,memory.shape[1])
  for layer in self.temporal:h=layer(h,mem,tgt_mask=causal,tgt_key_padding_mask=apad,memory_key_padding_mask=tpad)
  return self.final_norm(h)
 def frame_logits(self,context,codes):
  shape=context.shape[:-1];h=torch.tanh(self.depth_init(context.reshape(-1,self.dim)));inp=self.depth_start[None].expand(len(h),-1);out=[]
  for k in range(self.books):
   h=self.depth(inp,h);q=self.code_queries[k](h);logits=F.linear(q,self.code_embeddings[k].weight[:2048],self.code_bias[k])/16.;out.append(logits.reshape(*shape,2048));inp=self.code_embeddings[k](codes[...,k].reshape(-1))
  return out
 def losses(self,codes,memory,aligned,domains,audio_lengths,text_lengths,target_log_durations):
  context=self.context(codes,memory,aligned,domains,audio_lengths,text_lengths);logits=self.frame_logits(context,codes);mask=mask_for(audio_lengths,codes.shape[1]);parts=[];accuracy=[]
  for k,z in enumerate(logits):
   ce=F.cross_entropy(z.reshape(-1,2048).float(),codes[:,:,k].reshape(-1),reduction='none',label_smoothing=.01).reshape(mask.shape);parts.append((ce*mask).sum()/mask.sum());accuracy.append(((z.argmax(-1)==codes[:,:,k])*mask).sum()/mask.sum())
  weights=context.new_tensor([1]+[.5]*(self.books-1));codec=(torch.stack(parts)*weights).sum()/weights.sum();tm=mask_for(text_lengths,memory.shape[1]);dur=((self.log_durations(memory,domains)-target_log_durations).square()*tm).sum()/tm.sum()
  return codec+.1*dur,dict(codec=codec,duration=dur,first_code_accuracy=accuracy[0],all_code_accuracy=torch.stack(accuracy).mean())
 @torch.no_grad()
 def generate(self,ids,domain=0,seed=21,temperature=0.,top_k=30,max_frames=500):
  self.eval();lengths=torch.tensor([ids.shape[1]],device=ids.device);memory,mu=self.stem(ids,lengths);domains=torch.tensor([domain],device=ids.device)
  durations=self.log_durations(memory,domains)[0].float().clamp(0,math.log(100)).exp();frames=math.ceil(float(durations.sum())*256/1920)
  if not 2<=frames<=max_frames:raise ValueError(f'Predicted {frames} codec frames outside2..{max_frames}; no truncation')
  aligned=coarse_alignment(mu[0],durations,frames)[None];codes=torch.zeros(1,frames,self.books,dtype=torch.long,device=ids.device);rng=torch.Generator(device=ids.device).manual_seed(seed)
  for t in range(frames):
   h=self.context(codes[:,:t+1],memory,aligned[:,:t+1],domains,torch.tensor([t+1],device=ids.device),lengths)[:,-1];h=torch.tanh(self.depth_init(h));inp=self.depth_start[None]
   for k in range(self.books):
    h=self.depth(inp,h);logits=F.linear(self.code_queries[k](h),self.code_embeddings[k].weight[:2048],self.code_bias[k])/16.
    if temperature<=0:token=logits.argmax(-1)
    else:
     values,indices=(logits/temperature).topk(top_k,dim=-1);pick=torch.multinomial(values.softmax(-1),1,generator=rng);token=indices.gather(-1,pick)[:,0]
    codes[:,t,k]=token;inp=self.code_embeddings[k](token)
  return codes.transpose(1,2),dict(frames=frames,durations_fine_frames=durations.cpu().tolist(),seed=seed,temperature=temperature,top_k=top_k)
