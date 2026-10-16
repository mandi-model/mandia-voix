"""Matched small/wide vocoder trial; all full states immutable in Google Drive."""
from pathlib import Path
import sys,json,hashlib,random,time,math,argparse,gc,re,unicodedata
from dataclasses import asdict
from functools import lru_cache
P=Path('/content/drive/MyDrive/IvoireTTS_v01/06_mandia_preparation_v1');assert (P.parent/'pipeline_state.json').is_file()
for folder in ['research-phoneme-v01','research-voice-quality-v02','research-phase-consistent-v01']:sys.path.insert(0,str(P/'code'/folder))
import torch
from mandia_voice import Config
from phase_vocoder import ConsistentVocoder
from context_vocoder import CORE,CONTEXT,crop,discriminator_loss
from natural_vocoder import widen,NaturalDiscriminator,faithful_reconstruction,natural_generator_loss
from corpus import sha,write_json
import checkpoint_store as store
parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=['small','wide'],required=True);args=parser.parse_args()
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;device=torch.device('cuda');seed=2026101605;steps=4000;sampler=random.Random(seed)
torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
D=P/'naturalness-width-v01';R=D/args.branch;S=R/'checkpoints';S.mkdir(parents=True,exist_ok=True)
F=P/'pronunciation-phoneme-v01/features';ready=json.loads((F/'READY.json').read_text());assert sha(F/'rows.json')==ready['rows_sha256'];rows=json.loads((F/'rows.json').read_text())
total=CORE+2*CONTEXT
def long_enough(r):return int(r['seconds']*24000)>=total*256+512
siwis=[r for r in rows if r['domain']=='siwis' and long_enough(r)]
split=json.loads((P/'voice-adapter-v01/SPLITS.json').read_text())
training={'siwis':[r for r in siwis if r['split']=='train'],'ivoire':[r for r in split['train'] if long_enough(r)]}
validation={'siwis':sorted([r for r in siwis if r['split']=='validation'],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:20],'ivoire':[r for r in split['validation'] if long_enough(r)]}
assert len(training['ivoire'])>=20 and len(validation['ivoire'])>=4
fresh=json.loads((P/'code/naturalness-reserved-texts-v01.json').read_text())
def norm(text):return ' '.join(re.findall(r'[^\W_]+',unicodedata.normalize('NFKC',text).casefold()))
prior=[]
for file in [P/'pronunciation-phoneme-v01/fresh-test-texts.json',P/'voice-adapter-v01/reserved-test-texts.json']:prior+=json.loads(file.read_text())
assert not {norm(r['text']) for r in fresh}&{norm(r['text']) for r in rows+prior}
write_json(D/'reserved-test-texts.json',fresh)
@lru_cache(maxsize=128)
def shard(path,digest):assert sha(Path(path))==digest;return torch.load(path,map_location='cpu',weights_only=True)
def load(r):return shard(r['feature'],r['feature_sha256'])[r['feature_index']]
initial=P/'phase-consistent-v01/vocoder/best.pt';assert sha(initial)=='b0a0ad7010176176ef4fed0cc8a83d444bd8533f741d99988fd6255d7b584f43'
state=torch.load(initial,map_location=device,weights_only=True);base=ConsistentVocoder(Config()).to(device);base.load_state_dict(state['model'])
model=widen(base) if args.branch=='wide' else base;config=model.config
probe=load(validation['siwis'][0])['mel'].float().to(device)[None]
with torch.inference_mode():
    before=base(probe);after=model(probe);max_transfer_error=float((before-after).abs().max());relative_transfer_RMSE=float((before-after).square().mean().sqrt()/before.square().mean().sqrt())
assert max_transfer_error<1e-3 and relative_transfer_RMSE<1e-3,'Widening did not preserve the initial waveform'
del probe,before,after
if args.branch=='wide':del base
# Discriminator initialization is identical for both capacities, including new MSDs.
torch.manual_seed(2026101699);torch.cuda.manual_seed_all(2026101699);disc=NaturalDiscriminator().to(device);disc.previous.load_state_dict(state['discriminator']);del state
torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.8,.99));d_optimizer=torch.optim.AdamW(disc.parameters(),lr=5e-5,betas=(.8,.99))
settings={'branch':args.branch,'steps':steps,'seed':seed,'config':asdict(config),'batch_size':6,'initial_phase_vocoder_sha256':sha(initial),'initial_transfer_max_wave_error':max_transfer_error,'initial_transfer_relative_RMSE':relative_transfer_RMSE,'width_transfer':'Net2Wider-inspired duplication; LayerNorm statistics preserved; zero-sum input-column perturbations break symmetry without changing the initial function within numerical tolerance','train_counts':{k:len(v) for k,v in training.items()},'train_hours':{k:sum(r['seconds'] for r in v)/3600 for k,v in training.items()},'Ivoire_sampling_probability':.1,'Ivoire_source':'yt_6AZgZs5aaaQ; only49 previous training clips, with temporal validation/guard preserved; one real excerpt approved, not all clips','validation_ids':{k:[r['id'] for r in v] for k,v in validation.items()},'rows_sha256':ready['rows_sha256'],'core_frames':CORE,'context_frames_each_side':CONTEXT,'consistency_weight':.25,'reconstruction':'Previous multiresolution STFT+5mel, plus .5 log-magnitude/convergence FFT2048 loss','selection':'Mean of domain means (SIWIS and reference Ivoire) for reconstruction+.25consistency; each domain reconstruction must remain <=110% of initialization. Stepzero included. No synthesized test used.','optimizer':'AdamW .8/.99, 5e-5, 100-step warmup cosine to25%; identical across branches','new_training_only_adversary':'Three reduced multiscale waveform discriminators plus warmed own STFT/MPD; concept HiFi-GAN, not reproduction','model_parameters':sum(p.numel() for p in model.parameters()),'training_discriminator_parameters':sum(p.numel() for p in disc.parameters()),'trainer_sha256':sha(Path(__file__)),'code_sha256':{file.name:sha(file) for file in Path(__file__).parent.glob('*.py')},'frozen_dependencies':{folder:{file.name:sha(file) for file in (P/'code'/folder).glob('*.py')} for folder in ['research-phoneme-v01','research-voice-quality-v02','research-phase-consistent-v01']},'checkpoint_strategy':'Immutable UUID-tagged per-step states with individual SHA256 receipts; no rolling binary rename','fresh_test_texts_sha256':sha(P/'code/naturalness-reserved-texts-v01.json'),'human_naturalness_validated':False}
signature=hashlib.sha256(json.dumps({k:v for k,v in settings.items() if not k.startswith('initial_transfer_')},sort_keys=True).encode()).hexdigest();write_json(R/'PROTOCOL.json',dict(settings,signature=signature))
components={}
def validate():
    model.eval();domain_scores={};values={}
    with torch.inference_mode():
        for domain,records in validation.items():
            recon=[];cons=[]
            for row in records:
                mel,real=crop([load(row)],config,random.Random(77),False);full,inc=model.waveform_and_consistency(mel.to(device),CONTEXT,CONTEXT+CORE+1);fake=full[:,CONTEXT*256:(CONTEXT+CORE)*256];rec=faithful_reconstruction(fake,real.to(device),config);recon.append(float(rec));cons.append(float(inc))
            values[domain]={'reconstruction':sum(recon)/len(recon),'inconsistency_squared':sum(cons)/len(cons)};domain_scores[domain]=values[domain]['reconstruction']+.25*values[domain]['inconsistency_squared']
    components.clear();components.update(values);model.train();return sum(domain_scores.values())/len(domain_scores)
row,state=store.latest(S,signature);start=0;best_value=float('inf');previous_seconds=0
if state is not None:
    model.load_state_dict(state['model']);disc.load_state_dict(state['discriminator']);optimizer.load_state_dict(state['optimizer']);d_optimizer.load_state_dict(state['d_optimizer']);sampler.setstate(state['sampler_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);start=state['step'];best_value=state['best_validation'];previous_seconds=state['seconds'];del state;print('NATURALNESS_RESUMED',args.branch,start,flush=True)
def payload(step,value,seconds):return {'model':model.state_dict(),'discriminator':disc.state_dict(),'optimizer':optimizer.state_dict(),'d_optimizer':d_optimizer.state_dict(),'step':step,'validation':value,'validation_components':dict(components),'best_validation':best_value,'seconds':seconds,'signature':signature,'settings':settings,'sampler_rng':sampler.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
if start==0:
    value=validate();best_value=value;first=store.save(S,payload(0,value,0));write_json(R/'initial-validation.json',{'validation':value,'components':dict(components),'checkpoint':first});print('NATURALNESS_INITIAL',args.branch,value,'vocoder_params',settings['model_parameters'],flush=True)
initial_components=json.loads((R/'initial-validation.json').read_text())['components'];write_json(R/'STATUS.json',{'phase':'training','updates_target':steps,'immutable_checkpoints_in_Drive':True});started=time.monotonic()
for step in range(start+1,steps+1):
    records=[sampler.choice(training['ivoire' if sampler.random()<.1 else 'siwis']) for _ in range(6)];mel,real=crop([load(r) for r in records],config,sampler);mel=mel.to(device);real=real.to(device)
    rate=5e-5*min(1.,step/100)*(.25+.75*.5*(1+math.cos(math.pi*step/steps)))
    for opt in [optimizer,d_optimizer]:
        for group in opt.param_groups:group['lr']=rate
    optimizer.zero_grad(set_to_none=True);full,consistency=model.waveform_and_consistency(mel,CONTEXT,CONTEXT+CORE+1);fake=full[:,CONTEXT*256:(CONTEXT+CORE)*256]
    if step>=100:
        disc.requires_grad_(True);d_optimizer.zero_grad(set_to_none=True);lossd=discriminator_loss(fake,real,disc);assert torch.isfinite(lossd);lossd.backward();torch.nn.utils.clip_grad_norm_(disc.parameters(),5.,error_if_nonfinite=True);d_optimizer.step()
    disc.requires_grad_(False);loss=natural_generator_loss(fake,real,disc,config,consistency,step>=100);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    if step%50==0:print('NATURALNESS_TRAIN',args.branch,step,steps,float(loss.detach()),flush=True)
    if step%500==0 or step==steps:
        value=validate();admitted=all(components[k]['reconstruction']<=initial_components[k]['reconstruction']*1.10 for k in components);best_value=min(best_value,value) if admitted else best_value;seconds=previous_seconds+time.monotonic()-started;receipt=store.save(S,payload(step,value,seconds),admitted);best_row,_=store.best(S,signature)
        write_json(R/'progress.json',{'step':step,'target':steps,'validation':value,'components':dict(components),'admitted':admitted,'best_validation':best_value,'best_step':best_row['step'],'seconds':seconds,'latest_snapshot':receipt,'best_snapshot':best_row,'human_naturalness_validated':False});print('NATURALNESS_IMMUTABLE_DRIVE_CHECKPOINT',args.branch,step,value,'admitted',admitted,flush=True)
best_row,best_state=store.best(S,signature);write_json(R/'SELECTED-CHECKPOINT.json',best_row);write_json(R/'STATUS.json',{'phase':'training_complete','updates':steps,'selected_step':best_row['step'],'immutable_checkpoints_in_Drive':True,'parameter_count':settings['model_parameters'],'new_text_tests_pending':True});print('NATURALNESS_TRAINING_COMPLETE',args.branch,'selected',best_row['step'],flush=True)
