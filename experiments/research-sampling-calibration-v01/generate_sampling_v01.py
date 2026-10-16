"""Only calibrate initial noise amplitude; preserve model weights and durations."""
from pathlib import Path
import sys,json,hashlib,math,argparse,re,unicodedata
P=Path('/content/drive/MyDrive/IvoireTTS_v01/06_mandia_preparation_v1');assert (P.parent/'pipeline_state.json').is_file()
for folder in ['research-phoneme-v01','research-naturalness-width-v01','research-phase-consistent-v01','research-voice-cln-v01']:sys.path.insert(0,str(P/'code'/folder))
import torch,soundfile as sf
from mandia_voice import Acoustic,Config,tokenize,mask_for
from bounded_voice import integrate
from phase_vocoder import ConsistentVocoder
from corpus import sha,write_json
import checkpoint_store as store
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['development','fresh'],required=True);stage=parser.parse_args().stage
torch.set_num_threads(2);device=torch.device('cuda');D=P/'sampling-calibration-v01';D.mkdir(exist_ok=True);E=D/stage;E.mkdir(exist_ok=True)
F=P/'pronunciation-phoneme-v01/features';rows=json.loads((F/'rows.json').read_text());lookup={r['id']:r for r in rows};stats=json.loads((F/'normalization.json').read_text());mean=torch.tensor(stats['mean'],device=device);std=torch.tensor(stats['std'],device=device)
initial=P/'pronunciation-phoneme-v01/siwis/best.pt';assert sha(initial)=='b65fe825341d27bf3a088b5079b29b9f24a6f410b584f15a91143ee73002e758';base=Acoustic(Config()).to(device).eval();base.load_state_dict(torch.load(initial,map_location=device,weights_only=True)['model']);R=P/'naturalness-width-v01/small';vocrow=json.loads((R/'SELECTED-CHECKPOINT.json').read_text());assert vocrow['sha256']=='7cc01ded07cc1e2aa67403250c8dbc2fb4477f988e918ae4a34bb33aac76cf52';state=store.read(R/'checkpoints',vocrow,device);voc=ConsistentVocoder(Config(**state['settings']['config'])).to(device).eval();voc.load_state_dict(state['model']);del state
fresh=json.loads((P/'code/sampling-fresh-texts-v01.json').read_text());known=list(rows)
for path in [P/'pronunciation-phoneme-v01/fresh-test-texts.json',P/'voice-adapter-v01/reserved-test-texts.json',P/'naturalness-width-v01/reserved-test-texts.json',P/'voice-cln-v01/reserved-test-texts.json']:known+=json.loads(path.read_text())
def norm(text):return ' '.join(re.findall(r'[^\W_]+',unicodedata.normalize('NFKC',text).casefold()))
assert len(fresh)==8 and not {norm(r['text']) for r in fresh}&{norm(r['text']) for r in known}
if stage=='development':
 write_json(D/'reserved-fresh-texts.json',fresh);diag=json.loads((P/'acoustic-diagnostic-v01/PROTOCOL.json').read_text());records=[lookup[k] for k in diag['strata_ids']['development']];conditions=[('t1000',1.),('t0667',.667),('t0500',.5)]
 protocol={'no_weight_update':True,'initial_acoustic_sha256':sha(initial),'vocoder_snapshot':vocrow,'parameters_TTS':9368355,'noise_amplitudes':[1.,.667,.5],'ODE':'Euler32, same fixed duration predictor','development_ids':[r['id'] for r in records],'development_seeds':[8800,9900],'fresh_seeds':[12000,13000],'fresh_prompt_sha256':sha(P/'code/sampling-fresh-texts-v01.json'),'selection':'On8 previous development texts, aggregate both seeds: candidate must improve spoken word-edit count by >=2, not worsen spoken CER or repeated trigram excess. Then minimum word edits, character edits, closest amplitude to1. Baseline included. No fresh result selects amplitude.','source_sha256':{f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},'reference':'https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/cli.py','reference_scope':'Official CLI exposes temperature and defaults to.667; this is motivation, not evidence that.667 improves this own model.','limits':'Development reused from prior diagnostic;8 fresh texts and2 seeds are still a small sample. Less noise may flatten prosody. No accent or naturalness validation, no new pretrained weights.'};write_json(D/'PROTOCOL.json',protocol)
else:
 records=fresh;selected=json.loads((D/'SELECTED-SAMPLING.json').read_text());conditions=[('baseline',1.),('selected',selected['temperature'])];protocol=json.loads((D/'PROTOCOL.json').read_text());assert sha(initial)==protocol['initial_acoustic_sha256']
@torch.no_grad()
def sample(text,seed,temperature):
 tokens=torch.tensor([tokenize(text)],device=device);nt=torch.tensor([tokens.shape[1]],device=device);sp=torch.zeros(1,dtype=torch.long,device=device);means,ld,mask=base.encode(tokens,nt,sp);dur=torch.ceil(ld.clamp(0,math.log(100)).exp()).long()*mask;nf=dur.sum(-1);assert int(nf.max())<=5000;aligned=means.new_zeros(1,int(nf[0]),80);aligned[0]=torch.repeat_interleave(means[0],dur[0],dim=0);fm=mask_for(nf);rng=torch.Generator(device=device).manual_seed(seed);x=torch.randn(aligned.shape,device=device,generator=rng)*temperature*fm[...,None];mel=integrate(base.velocity,x,aligned,fm,sp,32,'euler')
 return mel,nf,tokens,nt,sp
items=[]
with torch.inference_mode():
 for condition,temperature in conditions:
  for i,row in enumerate(records):
   for j,seed0 in enumerate([8800,9900] if stage=='development' else [12000,13000]):
    seed=seed0+i;mel,nf,text,nt,sp=sample(row['text'],seed,temperature)
    if stage=='development' and condition=='t1000' and i==0 and j==0:
     original,n=base.synthesize(text,nt,steps=32,seed=seed,speaker_ids=sp);torch.testing.assert_close(mel,original,rtol=0,atol=0);assert torch.equal(nf,n);print('SAMPLING_BASELINE_BITWISE_MATCH',flush=True)
    wave=voc(mel*std+mean)[0];assert torch.isfinite(wave).all();peak=float(wave.abs().max());rms=float(wave.square().mean().sqrt())
    for listening in [False,True]:
     scale=min(.06/max(rms,1e-6),.95/max(peak,1e-6)) if listening else min(1.,.95/max(peak,1e-6));file=E/(condition+'-'+row['id']+'-seed'+str(j+1)+('-listening' if listening else '')+'.wav');sf.write(file,wave.cpu().numpy()*scale,24000,subtype='PCM_16');items.append({'id':row['id'],'text':row['text'],'condition':condition,'temperature':temperature,'seed':seed,'seed_index':j+1,'kind':'listening' if listening else 'raw','file':str(file),'sha256':sha(file),'seconds':len(wave)/24000,'frames':int(nf[0]),'applied_scale':scale,'normalized_mel_temporal_std_mean':float(mel[0].std(0).mean())})
    print('SAMPLING_AUDIO_SAVED',stage,condition,row['id'],j+1,flush=True)
write_json(E/'AUDIO-RECEIPTS.json',{'items':items,'protocol':protocol,'no_weight_update':True});print('SAMPLING_AUDIO_COMPLETE',stage,flush=True)
