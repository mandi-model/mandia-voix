"""Standalone inference for Mandia Voix 9M (experimental French reference)."""
import argparse,hashlib,json,math,sys,wave
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent/"src"))
from mandia_voice import Acoustic,Config,mask_for
from phase_vocoder import ConsistentVocoder
from phoneme_frontend import tokenize

def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def load_models(directory,device="cpu"):
 directory=Path(directory);manifest=json.loads((Path(__file__).parent/"WEIGHTS.json").read_text())
 for item in manifest["files"]:
  if digest(directory/item["name"])!=item["sha256"]:raise ValueError("Weight integrity failure: "+item["name"])
 a=torch.load(directory/"acoustic.pt",map_location=device,weights_only=True)
 v=torch.load(directory/"vocoder.pt",map_location=device,weights_only=True)
 acoustic=Acoustic(Config(**a["config"])).to(device).eval()
 vocoder=ConsistentVocoder(Config(**v["config"])).to(device).eval()
 acoustic.load_state_dict(a["model"],strict=True);vocoder.load_state_dict(v["model"],strict=True)
 stats=json.loads((Path(__file__).parent/"normalization.json").read_text())
 return acoustic,vocoder,torch.tensor(stats["mean"],device=device),torch.tensor(stats["std"],device=device)
@torch.inference_mode()
def infer_tokens(acoustic,vocoder,mean,std,token_ids,seed=12000,temperature=.5,steps=32):
 if not 0<=temperature<=2 or not 1<=steps<=128:raise ValueError("Invalid ODE settings")
 device=mean.device;tokens=torch.tensor([token_ids],device=device,dtype=torch.long)
 if not 1<=tokens.numel()<=500:raise ValueError("Use a short sentence (maximum500 phonetic symbols)")
 nt=torch.tensor([tokens.shape[1]],device=device);speaker=torch.zeros(1,dtype=torch.long,device=device)
 means,ld,mask=acoustic.encode(tokens,nt,speaker)
 durations=torch.ceil(ld.clamp(0,math.log(100)).exp()).long()*mask;nf=durations.sum(-1)
 if not 2<=int(nf[0])<=5000:raise ValueError("Frame budget exceeded; split the text")
 aligned=means.new_zeros(1,int(nf[0]),acoustic.config.mel)
 aligned[0]=torch.repeat_interleave(means[0],durations[0],dim=0);fm=mask_for(nf)
 rng=torch.Generator(device=device).manual_seed(seed)
 x=torch.randn(aligned.shape,device=device,generator=rng)*temperature*fm[...,None]
 for i in range(steps):
  time=x.new_full((1,),i/steps)
  x=(x+acoustic.velocity(x,aligned,time,fm,speaker)/steps)*fm[...,None]
 audio=vocoder(x*std+mean)[0]
 if not torch.isfinite(audio).all():raise ValueError("Non-finite audio")
 peak=float(audio.abs().max());rms=float(audio.square().mean().sqrt())
 return audio.cpu()*min(.06/max(rms,1e-6),.95/max(peak,1e-6))
def main():
 p=argparse.ArgumentParser();p.add_argument("--text",required=True);p.add_argument("--weights",default="weights")
 p.add_argument("--output",default="voice.wav");p.add_argument("--device",default="cpu");p.add_argument("--seed",type=int,default=12000)
 a=p.parse_args();torch.set_num_threads(2)
 if not a.text.strip():raise ValueError("Empty text")
 models=load_models(a.weights,a.device);audio=infer_tokens(*models,tokenize(a.text),seed=a.seed)
 pcm=(audio.clamp(-1,1).numpy()*32767).round().astype("<i2")
 with wave.open(a.output,"wb") as f:
  f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(pcm.tobytes())
 print(a.output)
if __name__=="__main__":main()

