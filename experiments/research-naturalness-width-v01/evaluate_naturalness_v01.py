"""Matched fresh-text and real-mel controls. Human listening remains necessary."""
from pathlib import Path
import sys,json,time,statistics,hashlib
P=Path('/content/drive/MyDrive/IvoireTTS_v01/06_mandia_preparation_v1');assert (P.parent/'pipeline_state.json').is_file()
for folder in ['research-phoneme-v01','research-voice-quality-v02','research-phase-consistent-v01']:sys.path.insert(0,str(P/'code'/folder))
import torch,soundfile as sf
from mandia_voice import Config,Acoustic,tokenize
from phase_vocoder import ConsistentVocoder
from corpus import sha,write_json
import checkpoint_store as store
torch.set_num_threads(2);device=torch.device('cuda');D=P/'naturalness-width-v01';E=D/'evaluation';E.mkdir(exist_ok=True)
F=P/'pronunciation-phoneme-v01/features';rows=json.loads((F/'rows.json').read_text());lookup={r['id']:r for r in rows};stats=json.loads((F/'normalization.json').read_text());mean=torch.tensor(stats['mean'],device=device);std=torch.tensor(stats['std'],device=device)
fresh=[dict(r,kind='fresh') for r in json.loads((D/'reserved-test-texts.json').read_text())]
development={'id':'p-new-004','text':'Peux-tu fermer doucement la porte et laisser la fenêtre ouverte ?','kind':'development'};prompts=fresh+[development]
acoustic_file=P/'pronunciation-phoneme-v01/siwis/best.pt';assert sha(acoustic_file)=='b65fe825341d27bf3a088b5079b29b9f24a6f410b584f15a91143ee73002e758'
acoustic=Acoustic(Config()).to(device).eval();acoustic.load_state_dict(torch.load(acoustic_file,map_location=device,weights_only=True)['model'])
mel_file=E/'fixed-test-mels.pt'
if mel_file.exists():
 assert sha(mel_file)==json.loads(mel_file.with_suffix('.json').read_text())['sha256'];mels=torch.load(mel_file,map_location='cpu',weights_only=True)
else:
 mels={}
 with torch.inference_mode():
  for i,r in enumerate(prompts):
   text=torch.tensor([tokenize(r['text'])],device=device);nt=torch.tensor([text.shape[1]],device=device);mel,n=acoustic.synthesize(text,nt,steps=32,seed=903 if r['kind']=='development' else 1500+i,speaker_ids=torch.zeros(1,dtype=torch.long,device=device));mels[r['id']]=(mel*std+mean).cpu()
 with mel_file.open('xb') as stream:torch.save(mels,stream)
 write_json(mel_file.with_suffix('.json'),{'sha256':sha(mel_file),'acoustic_sha256':sha(acoustic_file),'ODE_steps':32,'seeds':'1500+i for6 fresh texts;903 for prior development sentence','prompts':prompts})
del acoustic;torch.cuda.empty_cache();items=[];latency={};selected={}
def output(wave,name,r,condition,kind,matched=False):
 assert torch.isfinite(wave).all();peak=float(wave.abs().max());rms=float(wave.square().mean().sqrt());scale=min(.06/max(rms,1e-6),.95/max(peak,1e-6)) if matched else min(1.,.95/max(peak,1e-6));file=E/(name+'.wav');sf.write(file,wave.cpu().numpy()*scale,24000,subtype='PCM_16')
 return dict(id=r['id'],text=r['text'],condition=condition,kind=kind,file=str(file),sha256=sha(file),seconds=len(wave)/24000,peak_before_scale=peak,RMS_before_scale=rms,applied_scale=scale,matched_listening_RMS_target=.06 if matched else None)
controls=json.loads((P/'pronunciation-phoneme-v01/evaluation/AUDIO-RECEIPTS.json').read_text())['items'];real_controls=[r for r in controls if r['condition'].endswith('/real')]
with torch.inference_mode():
 for branch in ['phase_initial','small','wide']:
  if branch=='phase_initial':
   file=P/'phase-consistent-v01/vocoder/best.pt';assert sha(file)=='b0a0ad7010176176ef4fed0cc8a83d444bd8533f741d99988fd6255d7b584f43';state=torch.load(file,map_location=device,weights_only=True);digest=sha(file)
  else:
   root=D/branch;receipt=json.loads((root/'SELECTED-CHECKPOINT.json').read_text());state=store.read(root/'checkpoints',receipt,device=device);digest=receipt['sha256']
  config=Config(**state['settings']['config']);voc=ConsistentVocoder(config).to(device).eval();voc.load_state_dict(state['model']);selected[branch]={'step':state['step'],'sha256':digest,'vocoder_parameters':sum(p.numel() for p in voc.parameters()),'total_TTS_parameters':8423841+sum(p.numel() for p in voc.parameters())}
  probe=mels[development['id']].to(device)
  for _ in range(3):voc(probe)
  torch.cuda.synchronize();times=[]
  for _ in range(10):
   start=time.perf_counter();wave=voc(probe);torch.cuda.synchronize();times.append(time.perf_counter()-start)
  latency[branch]={'vocoder_only_median_seconds':statistics.median(times),'vocoder_only_median_RTF':statistics.median(times)/(wave.shape[-1]/24000),'repetitions':10,'warmups':3,'audio_seconds':wave.shape[-1]/24000,'gpu':torch.cuda.get_device_name(0),'excludes_acoustic_ASR_IO_and_model_load':True,'individual_seconds':times}
  for r in prompts:
   wave=voc(mels[r['id']].to(device))[0];kind=r['kind'];item=output(wave,branch+'-'+r['id'],r,kind+'/'+branch,kind);item['checkpoint_sha256']=digest;items.append(item)
   matched=output(wave,branch+'-'+r['id']+'-listening-matched',r,'listening/'+branch,'listening',True);items.append(matched);print('NATURALNESS_AUDIO_SAVED',branch,r['id'],flush=True)
  for r in real_controls:
   row=lookup[r['id']];assert sha(Path(row['feature']))==row['feature_sha256'];sample=torch.load(row['feature'],map_location='cpu',weights_only=True)[row['feature_index']];wave=voc(sample['mel'].float().to(device)[None])[0];domain=row['domain'];items.append(output(wave,branch+'-realmel-'+row['id'],r,domain+'/'+branch+'_real_mel','control'))
  del voc,state;torch.cuda.empty_cache()
 for r in real_controls:
  assert sha(Path(r['file']))==r['sha256'];items.append(dict(r,kind='control'))
write_json(E/'AUDIO-RECEIPTS.json',{'items':items,'selected':selected,'vocoder_latency':latency,'fresh_texts':6,'development_texts':1,'base_acoustic_sha256':sha(acoustic_file),'same_test_mels_sha256':sha(mel_file),'matched_study':'Small and wide share initial function within measured numerical tolerance, data/sampler seed, loss, discriminator initialization, optimizer,4000 updates, validation criteria. Widening changes capacity and symmetry. Both new branches also differ in recipe/data mixture from the earlier phase baseline. One seed; not a general superiority study.','human_naturalness_validated':False,'synthetic_Ivorian_accent_validated':False,'release_ready':False})
print('NATURALNESS_PAIRED_AUDIO_COMPLETE',json.dumps(selected),flush=True)
