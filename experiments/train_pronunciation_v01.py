"""Two-stage own-weight phonetic TTS study. Full Drive checkpoints; fixed fresh-text tests.
SIWIS pretraining then IvoireTTS adaptation with 20% SIWIS replay. Not a matched architecture ablation.
"""
from pathlib import Path
import sys,json,random,hashlib,time,math,argparse
from dataclasses import asdict
from functools import lru_cache
P=Path('/content/drive/MyDrive/IvoireTTS_v01/06_mandia_preparation_v1');assert (P.parent/'pipeline_state.json').is_file()
C=P/'code/research-phoneme-v01';sys.path.insert(0,str(C))
import torch,numpy as np,soundfile as sf
from mandia_voice import Config,Acoustic,FourierVocoder,tokenize,mask_for,parameter_counts
from corpus import sha,write_json
config=Config();torch.set_num_threads(2)
D=P/'pronunciation-phoneme-v01';F=D/'features'
ready=json.loads((F/'READY.json').read_text());rows=json.loads((F/'rows.json').read_text());stats=json.loads((F/'normalization.json').read_text())
assert sha(F/'rows.json')==ready['rows_sha256'] and sha(F/'normalization.json')==ready['normalization_sha256']
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['siwis','ivoire'],required=True);parser.add_argument('--steps',type=int,required=True);args=parser.parse_args()
R=D/args.stage;R.mkdir(exist_ok=True)
seed=20261016;random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
device=torch.device('cuda');torch.backends.cudnn.benchmark=False
model=Acoustic(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4 if args.stage=='siwis' else 1e-4,betas=(.9,.99))
sampler=random.Random(seed)
training={domain:[r for r in rows if r['split']=='train' and r['domain']==domain] for domain in ['siwis','ivoire']}
validation=[]
for domain,n in ([('siwis',24)] if args.stage=='siwis' else [('siwis',16),('ivoire',8)]):
 validation += sorted([r for r in rows if r['split']=='validation' and r['domain']==domain],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:n]
initial=D/'siwis/best.pt' if args.stage=='ivoire' else None
settings={'stage':args.stage,'steps':args.steps,'seed':seed,'batch_size':8,'base_lr':4e-4 if args.stage=='siwis' else 1e-4,'config':asdict(config),'rows_sha256':ready['rows_sha256'],'normalization_sha256':ready['normalization_sha256'],'code':{p.name:sha(p) for p in C.glob('*.py')},'trainer_sha256':sha(Path(__file__)),'initial_checkpoint_sha256':sha(initial) if initial else None,'optimizer':'AdamW betas .9/.99; 500-update warmup; cosine to 25% base rate','Ivoire_initialization':'copy the SIWIS speaker embedding into speaker index 1, then adapt all acoustic weights; optimizer reset','replay_fraction':.2 if args.stage=='ivoire' else 0,'validation_ids':[r['id'] for r in validation],'fresh_text_not_used_for_selection':True,'validation_alignment_prior':'fixed zero positional prior at all checkpoints; no changing validation objective','external_pretrained_TTS_weights':False,'parameter_counts':parameter_counts(config)}
signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
write_json(R/'PROTOCOL.json',settings)
write_json(R/'STATUS.json',{'phase':'training_or_resume_starting','steps':args.steps,'checkpoints_in_Drive':True})
@lru_cache(maxsize=128)
def shard(path,digest):
 file=Path(path);assert sha(file)==digest
 return torch.load(file,map_location='cpu',weights_only=True)
def load(r):
 v=shard(r['feature'],r['feature_sha256'])[r['feature_index']]
 return {'tokens':v['tokens'],'mel':v['mel'].float(),'speaker_index':v['speaker_index']}
def batch(records):
 samples=[load(r) for r in records];nt=torch.tensor([len(x['tokens']) for x in samples],device=device);nf=torch.tensor([len(x['mel']) for x in samples],device=device)
 text=torch.zeros(len(samples),int(nt.max()),dtype=torch.long,device=device);mel=torch.zeros(len(samples),int(nf.max()),config.mel,device=device)
 for i,s in enumerate(samples):text[i,:nt[i]]=s['tokens'].to(device);mel[i,:nf[i]]=(s['mel'].to(device)-mean)/std
 return text,nt,mel,nf,torch.tensor([s['speaker_index'] for s in samples],device=device)
mean=torch.tensor(stats['mean'],device=device);std=torch.tensor(stats['std'],device=device)
def save(path,payload):
 assert (P.parent/'pipeline_state.json').is_file(),'Drive lost; no fallback on Mac or ephemeral disk'
 temp=path.with_suffix('.tmp');torch.save(payload,temp);digest=sha(temp);temp.replace(path);assert sha(path)==digest
 write_json(path.with_suffix('.sha256.json'),{'sha256':digest,'file':path.name,'bytes':path.stat().st_size})
def read(path):
 assert sha(path)==json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
 state=torch.load(path,map_location=device,weights_only=True);return state
last=R/'last.pt';start=0;best=float('inf');elapsed_previous=0
if not last.exists() and (R/'previous.pt').exists():
 import shutil
 read(R/'previous.pt');shutil.copyfile(R/'previous.pt',last);shutil.copyfile(R/'previous.sha256.json',R/'last.sha256.json')
if last.exists():
 state=read(last);assert state['signature']==signature
 model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);sampler.setstate(state['sampler_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']]);start=state['step'];best=state['best_validation'];elapsed_previous=state['seconds']
 print('RESUMED_DRIVE',args.stage,start,flush=True)
elif initial:
 state=read(initial);model.load_state_dict(state['model'])
 with torch.no_grad():model.speaker.weight[1].copy_(model.speaker.weight[0])
 print('OWN_SIWIS_CHECKPOINT_LOADED_FOR_IVOIRE',sha(initial),flush=True)
def validate(step):
 model.eval();model.training_step=6000;values=[]
 with torch.inference_mode():
  for r in validation:
   text,nt,mel,nf,sp=batch([r]);parts=model.losses(text,nt,mel,nf,speaker_ids=sp,generator=torch.Generator(device=device).manual_seed(77));values.append(float(sum(parts.values())))
 model.train();return sum(values)/len(values)
def payload(step,value,seconds):
 return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'validation':value,'best_validation':best,'signature':signature,'settings':settings,'seconds':seconds,'sampler_rng':sampler.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
# Include step zero in best-checkpoint selection, unlike the old continuation study.
if start==0:
 value=validate(0);best=value;save(R/'best.pt',payload(0,value,0));write_json(R/'initial-validation.json',{'step':0,'validation':value});print('INITIAL_VALIDATION',args.stage,value,flush=True)
started=time.monotonic()
for step in range(start+1,args.steps+1):
 model.train();model.training_step=step+(18000 if args.stage=='ivoire' else 0)
 rate=settings['base_lr']*min(1.,step/500)*(.25+.75*.5*(1+math.cos(math.pi*step/args.steps)))
 for group in optimizer.param_groups:group['lr']=rate
 records=[]
 for _ in range(8):
  domain='ivoire' if args.stage=='ivoire' and sampler.random()>=.2 else 'siwis';records.append(sampler.choice(training[domain]))
 text,nt,mel,nf,sp=batch(records);optimizer.zero_grad(set_to_none=True)
 parts=model.losses(text,nt,mel,nf,speaker_ids=sp);loss=sum(parts.values());assert torch.isfinite(loss),'Non-finite loss; last Drive checkpoint retained'
 loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
 if step%50==0:print('PHONEME_TRAIN',args.stage,step,args.steps,float(loss.detach()),'lr',rate,flush=True)
 if step%250==0 or step==args.steps:
  value=validate(step);improved=value<best;best=min(best,value);seconds=elapsed_previous+time.monotonic()-started
  state=payload(step,value,seconds)
  if last.exists():last.replace(R/'previous.pt');last.with_suffix('.sha256.json').replace(R/'previous.sha256.json')
  save(last,state)
  if improved:save(R/'best.pt',state)
  write_json(R/'progress.json',{'step':step,'target':args.steps,'validation':value,'best_validation':best,'seconds':seconds,'gpu':torch.cuda.get_device_name(0),'checkpoint_sha256':sha(last),'generated_voice_validated':False})
  print('PHONEME_CHECKPOINT_DRIVE_VERIFIED',args.stage,step,value,flush=True)
write_json(R/'STATUS.json',{'phase':'training_complete','updates':args.steps,'best_step':read(R/'best.pt')['step'],'checkpoints_in_Drive':True,'fresh_text_evaluation_pending':True})
print('PHONEME_TRAINING_STAGE_COMPLETE',args.stage,flush=True)
