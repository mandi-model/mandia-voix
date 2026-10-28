"""Experimental text-to-Mimi inference. No target recording or true durations."""
from pathlib import Path
import argparse,hashlib,json,sys,urllib.request
import numpy as np
import torch
import soundfile as sf
from safetensors.torch import load_file
from transformers import MimiModel
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from mimi_model21 import MimiText21
from phoneme_frontend import tokenize

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--text',required=True);p.add_argument('--voice',choices=['fr','ci'],default='fr');p.add_argument('--output',type=Path,default=Path('output.wav'));p.add_argument('--cache',type=Path,default=Path('models'));p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--seed',type=int,default=321100);p.add_argument('--temperature',type=float,default=0.);args=p.parse_args()
 if not args.text.strip():p.error('Text is empty')
 if not 0<=args.temperature<=1:p.error('Temperature outside0..1')
 args.cache.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((ROOT/'WEIGHTS.json').read_text());entry=manifest['files'][args.voice];file=args.cache/entry['name']
 if not file.exists():
  tmp=file.with_suffix('.part');urllib.request.urlretrieve(entry['url'],tmp)
  if digest(tmp)!=entry['sha256']:raise ValueError('Downloaded weight checksum mismatch')
  tmp.replace(file)
 if digest(file)!=entry['sha256']:raise ValueError('Weight checksum mismatch')
 config=json.loads((ROOT/'model-config.json').read_text());model=MimiText21(**config).to(args.device).eval().requires_grad_(False);model.load_state_dict(load_file(str(file),device=args.device),strict=True)
 codec=MimiModel.from_pretrained('kyutai/mimi',revision=manifest['Mimi_revision'],cache_dir=str(args.cache/'huggingface'),trust_remote_code=False,attn_implementation='sdpa').float().to(args.device).eval().requires_grad_(False);codec.config.use_cache=False
 ids=torch.tensor([tokenize(args.text)],device=args.device)
 with torch.inference_mode():
  codes,trace=model.generate(ids,domain=int(args.voice=='ci'),seed=args.seed,temperature=args.temperature,top_k=30,max_frames=500);wave=codec.decode(codes).audio_values[0,0].float().cpu().numpy()
 if not np.isfinite(wave).all() or len(wave)<2400:raise ValueError('Invalid output')
 rms=float(np.sqrt(np.mean(wave*wave)));peak=float(np.max(np.abs(wave)))
 if rms<=1e-8:raise ValueError('Silent output')
 gain=min(.06/rms,.95/max(peak,1e-8));args.output.parent.mkdir(parents=True,exist_ok=True);sf.write(args.output,wave*gain,24000,subtype='PCM_16')
 record=dict(text=args.text,voice=args.voice,trace=trace,sample_rate=24000,samples=len(wave),gain=gain,weight_sha256=entry['sha256'],warning='Experimental. CI adaptation is less intelligible than N in the reserved-text ASR test; no validated general Ivorian accent.')
 args.output.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print(args.output)
if __name__=='__main__':main()
