"""Clean SIWIS studio/context-aware refinement. Full resumable Drive states.

Keeps the inference architecture unchanged. Validation uses fixed real-mel
reconstruction; human echo/naturalness judgment remains necessary.
"""
from pathlib import Path
import sys,json,random,time,math,hashlib
from functools import lru_cache
from dataclasses import asdict
P=Path('/content/drive/MyDrive/IvoireTTS_v01/06_mandia_preparation_v1');assert (P.parent/'pipeline_state.json').is_file()
sys.path.insert(0,str(P/'code/research-phoneme-v01'))
import torch
from mandia_voice import Config,FourierVocoder
from corpus import sha,write_json
from context_vocoder import CORE,CONTEXT,crop,center_wave,receptive_field,QualityDiscriminator,reconstruction,generator_loss,discriminator_loss
torch.set_num_threads(2);device=torch.device('cuda');config=Config()
D=P/'pronunciation-phoneme-v01';F=D/'features';R=P/'voice-quality-v02/vocoder';R.mkdir(parents=True,exist_ok=True)
ready=json.loads((F/'READY.json').read_text());assert sha(F/'rows.json')==ready['rows_sha256']
rows=json.loads((F/'rows.json').read_text());total=CORE+2*CONTEXT
accepted=[r for r in rows if r['domain']=='siwis' and int(r['seconds']*24000)>=total*config.hop+512]
train=[r for r in accepted if r['split']=='train']
validation=sorted([r for r in accepted if r['split']=='validation'],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:20]
assert len(train)>1000 and len(validation)>=10
seed=2026101602;steps=3000;sampler=random.Random(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
model=FourierVocoder(config).to(device);disc=QualityDiscriminator().to(device)
assert CONTEXT>(receptive_field(model)-1)//2+2
initial=P/'vocoder-clean-v01/vocoder/best.pt';assert sha(initial)=='be0958daf687ca0a05ffbb5da6ef8659befeaa74ea93cee65e2c2b2d9bc26c38'
state=torch.load(initial,map_location=device,weights_only=True);model.load_state_dict(state['model'])
# Reuse only the own previously trained spectral discriminator; periodic ones start random.
disc.spectral.load_state_dict(state['discriminator']);del state
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.8,.99))
d_optimizer=torch.optim.AdamW(disc.parameters(),lr=1e-4,betas=(.8,.99))
settings={'steps':steps,'seed':seed,'config':asdict(config),'batch_size':6,'initial_vocoder_sha256':sha(initial),'rows_sha256':ready['rows_sha256'],'train_clips':len(train),'train_hours':sum(r['seconds'] for r in train)/3600,'validation_ids':[r['id'] for r in validation],'core_frames':CORE,'context_each_side_frames':CONTEXT,'mel_receptive_field':receptive_field(model),'training_only_discriminator':'own reduced MPD periods 2,3,5,7,11 plus own existing magnitude STFT discriminator; HiFi-GAN inspired, not reproduction','training_corpus':'SIWIS studio train only, no YouTube audio in this vocoder refinement','selection':'fixed central real-mel reconstruction loss, step zero included; no synthesized fresh-text selection','human_echo_resolved':False,'trainer_sha256':sha(Path(__file__)),'context_module_sha256':sha(Path(__file__).with_name('context_vocoder.py')),'frozen_model_code_sha256':sha(P/'code/research-phoneme-v01/mandia_voice.py'),'optimizer':'AdamW .8/.99, generator 1e-4, discriminator 1e-4, 100-step warmup, cosine to 25%','adversarial_warmup':100,'inference_parameters':sum(p.numel() for p in model.parameters()),'training_discriminator_parameters':sum(p.numel() for p in disc.parameters())}
signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest();write_json(R/'PROTOCOL.json',settings)
@lru_cache(maxsize=128)
def shard(path,digest):
    assert sha(Path(path))==digest;return torch.load(path,map_location='cpu',weights_only=True)
def load(r):return shard(r['feature'],r['feature_sha256'])[r['feature_index']]
def save(path,state):
    assert (P.parent/'pipeline_state.json').is_file(),'Drive disconnected; stop without local fallback'
    temp=path.with_suffix('.tmp');torch.save(state,temp);digest=sha(temp);temp.replace(path);assert sha(path)==digest
    write_json(path.with_suffix('.sha256.json'),{'sha256':digest,'bytes':path.stat().st_size,'file':path.name})
def read(path):
    assert sha(path)==json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
    return torch.load(path,map_location=device,weights_only=True)
def validate():
    model.eval();values=[]
    with torch.inference_mode():
        for r in validation:
            mel,real=crop([load(r)],config,random.Random(77),False)
            values.append(float(reconstruction(center_wave(model,mel.to(device)),real.to(device),config)))
    model.train();return sum(values)/len(values)
last=R/'last.pt';start=0;best=float('inf');previous_seconds=0
if not last.exists() and (R/'previous.pt').exists():
    import shutil
    read(R/'previous.pt');shutil.copyfile(R/'previous.pt',last);shutil.copyfile(R/'previous.sha256.json',R/'last.sha256.json')
if last.exists():
    state=read(last);assert state['signature']==signature
    model.load_state_dict(state['model']);disc.load_state_dict(state['discriminator']);optimizer.load_state_dict(state['optimizer']);d_optimizer.load_state_dict(state['d_optimizer']);sampler.setstate(state['sampler_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']]);start=state['step'];best=state['best_validation'];previous_seconds=state['seconds']
def payload(step,value,seconds):return {'model':model.state_dict(),'discriminator':disc.state_dict(),'optimizer':optimizer.state_dict(),'d_optimizer':d_optimizer.state_dict(),'step':step,'validation':value,'best_validation':best,'seconds':seconds,'sampler_rng':sampler.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'signature':signature,'settings':settings}
if start==0:
    value=validate();best=value;save(R/'best.pt',payload(0,value,0));write_json(R/'initial-validation.json',{'step':0,'validation':value});print('CONTEXT_VOCODER_INITIAL',value,flush=True)
write_json(R/'STATUS.json',{'phase':'training','target':steps,'checkpoints_in_Drive':True});started=time.monotonic()
for step in range(start+1,steps+1):
    model.train();mel,real=crop([load(sampler.choice(train)) for _ in range(6)],config,sampler);mel=mel.to(device);real=real.to(device)
    rate=1e-4*min(1.,step/100)*(.25+.75*.5*(1+math.cos(math.pi*step/steps)))
    for opt in [optimizer,d_optimizer]:
        for group in opt.param_groups:group['lr']=rate
    optimizer.zero_grad(set_to_none=True);fake=center_wave(model,mel)
    if step>=100:
        disc.requires_grad_(True);d_optimizer.zero_grad(set_to_none=True);d_loss=discriminator_loss(fake,real,disc);assert torch.isfinite(d_loss);d_loss.backward();torch.nn.utils.clip_grad_norm_(disc.parameters(),5.,error_if_nonfinite=True);d_optimizer.step()
    disc.requires_grad_(False);loss=generator_loss(fake,real,disc,config,step>=100);assert torch.isfinite(loss)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    if step%50==0:print('CONTEXT_VOCODER_TRAIN',step,steps,float(loss.detach()),flush=True)
    if step%250==0 or step==steps:
        value=validate();improved=value<best;best=min(best,value);seconds=previous_seconds+time.monotonic()-started;state=payload(step,value,seconds)
        if last.exists():last.replace(R/'previous.pt');last.with_suffix('.sha256.json').replace(R/'previous.sha256.json')
        save(last,state)
        if improved:save(R/'best.pt',state)
        write_json(R/'progress.json',{'step':step,'target':steps,'validation':value,'best_validation':best,'seconds':seconds,'checkpoint_sha256':sha(last),'human_echo_resolved':False});print('CONTEXT_VOCODER_CHECKPOINT_DRIVE_VERIFIED',step,value,flush=True)
write_json(R/'STATUS.json',{'phase':'training_complete','updates':steps,'best_step':read(R/'best.pt')['step'],'checkpoints_in_Drive':True,'listening_pending':True});print('CONTEXT_VOCODER_TRAINING_COMPLETE',flush=True)
