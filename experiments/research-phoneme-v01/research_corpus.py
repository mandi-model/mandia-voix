"""Private automatically labelled conditional multi-speaker research corpus.
Anonymous cluster labels are hypotheses; never verified identities or human review.
"""
import collections,csv,hashlib,json,re,unicodedata
from pathlib import Path
from dataclasses import asdict
import numpy as np,soundfile as sf,torch
from mandia_voice import Config,tokenize,to_mel
from corpus import sha,write_json,read_wave

def norm(s):return ' '.join(re.findall(r"[^\W_]+",unicodedata.normalize('NFKC',s).casefold()))

def assemble(prepared,dest,config=Config()):
 prepared,dest=Path(prepared),Path(dest)
 assert str(dest.resolve()).startswith('/content/drive/MyDrive/')
 rows=[];centers={};seconds=collections.Counter()
 for result in sorted((prepared/'sessions').glob('*/result.json')):
  session=result.parent
  for sid,vec in json.loads((session/'speaker-centroids-refined.json').read_text()).items():centers[session.name+':'+sid]=np.asarray(vec)
  for row in json.loads((session/'segments.json').read_text()):
   if not row.get('automatic_candidate') or row.get('speaker_embedding_similarity',0)<.65 or row['end']-row['start']>12:continue
   
   try: token_count=len(tokenize(row['text_DRAFT']))
   except ValueError:continue
   if token_count>min(500,int((row['end']-row['start'])*24000//256)):continue
   key=row['source_id']+':'+str(row['speaker_local']);rows.append(dict(row,local_key=key));seconds[key]+=row['end']-row['start']
 groups=[]
 for key in sorted(seconds,key=lambda k:-seconds[k]):
  fits=[]
  for i,g in enumerate(groups):
   if any(k.split(':')[0]==key.split(':')[0] for k in g):continue
   sims=[float(centers[key]@centers[k]) for k in g]
   if min(sims)>=.78:fits.append((min(sims),i))
  fits.sort(reverse=True)
  if fits and (len(fits)==1 or fits[0][0]-fits[1][0]>=.05):groups[fits[0][1]].append(key)
  else:groups.append([key])
 reserved={norm(x['text']) for x in csv.DictReader((Path(__file__).parent/'phrases-test-reservees.csv').open())}
 unique=[];seen_text=set();seen_audio=set()
 for x in sorted(rows,key=lambda x:(x['source_id'],x['start'])):
  t=norm(x['text_DRAFT'])
  if t in seen_text or t in reserved or x['audio_sha256'] in seen_audio:continue
  seen_text.add(t);seen_audio.add(x['audio_sha256']);unique.append(x)
 retained=[]
 for group in groups:
  values=[x for x in unique if x['local_key'] in group]
  if len(values)<15 or sum(x['end']-x['start'] for x in values)<60:continue
  by_source=collections.defaultdict(list)
  for x in values:by_source[x['source_id']].append(x)
  split=[]
  # Same recording is split in time, with >=10 s gaps at both evaluation boundaries.
  # This tests held-out utterances; it does not establish cross-session generalization.
  for source,clips in by_source.items():
   clips.sort(key=lambda x:x['start']);n=len(clips)
   if n<5:continue
   nv=max(2,round(.1*n));nt=max(2,round(.1*n));cut1=n-nv-nt;cut2=n-nt
   boundary1=(clips[cut1-1]['end']+clips[cut1]['start'])/2 if cut1>0 else 0
   boundary2=(clips[cut2-1]['end']+clips[cut2]['start'])/2
   for i,x in enumerate(clips):
    if i<cut1 and x['end']<=boundary1-5:label='train'
    elif cut1<=i<cut2 and x['start']>=boundary1+5 and x['end']<=boundary2-5:label='validation'
    elif i>=cut2 and x['start']>=boundary2+5:label='test'
    else:continue
    split.append(dict(x,split=label))
  counts=collections.Counter(x['split'] for x in split)
  if counts['train']<8 or counts['validation']<2 or counts['test']<2:continue
  retained.append({'members':sorted(group),'rows':split,'train_seconds':sum(x['end']-x['start'] for x in split if x['split']=='train')})
 retained.sort(key=lambda x:-x['train_seconds']);out=[];voices=[]
 for i,g in enumerate(retained):
  sid='voice_'+hashlib.sha256('|'.join(g['members']).encode()).hexdigest()[:12]
  voices.append({'speaker_index':i,'anonymous_label':sid,'members':g['members'],'counts':dict(collections.Counter(x['split'] for x in g['rows'])),'identity_verified':False})
  out.extend(dict(x,speaker_index=i,speaker_id=sid,session_id=x['source_id'],text=x['text_DRAFT'],transcript_checked='automatic_medium_turbo_agreement',transcript_human_checked=False) for x in g['rows'])
 counts=collections.Counter(x['split'] for x in out);durations={k:sum(x['end']-x['start'] for x in out if x['split']==k) for k in ['train','validation','test']};errors=[]
 if counts['train']<80 or durations['train']<300:errors.append('Conditional pilot requires >=80 training clips and >=5 training minutes across retained voices')
 if counts['validation']<10 or counts['test']<10 or durations['validation']<30 or durations['test']<30:errors.append('Requires >=10 clips and >=30 seconds each in validation and test')
 if not voices:errors.append('No speaker hypothesis has >=8 train and >=2 validation/test clips')
 identity='multivoice_'+hashlib.sha256(json.dumps(voices,sort_keys=True).encode()).hexdigest()[:12];dataset=dest/identity
 report={'protocol':'automatic_conditional_multivoice_pilot_v0.2','research_pilot_ready':not errors,'errors':errors,'speaker_hypothesis':identity,'speaker_count':len(voices),'voices':voices,'counts':dict(counts),'seconds':durations,'destination':str(dataset),'split_protocol':'within_recording_temporal_utterance_holdout_minimum_10s_guard_gaps','cross_session_generalization_established':False,'human_reviewed':False,'voice_identity_verified':False,'accent_verified_by_listener':False,'music_free_verified_by_listener':False,'release_ready':False,'original_v0_1_audit_unchanged':True,'thresholds_are_private_pilot_operational_choices':True,'protocol_amendment':'Minimum training utterances revised from 100 to 80 for first bounded fit; >=5 train minutes, acoustic criteria and heldouts unchanged. Not a release-quality threshold.','data_authorization':'User affirmed proceeding; independent license document not provided'}
 dest.mkdir(parents=True,exist_ok=True);write_json(dest/'speaker-selection.json',report)
 if errors:return report
 dataset.mkdir(exist_ok=True)
 for x in out:
  source=prepared/x['audio'];assert sha(source)==x['audio_sha256'];info=sf.info(source);assert info.channels==1 and info.samplerate==24000
  x['audio_absolute']=str(source);assert len(tokenize(x['text']))<=500
 write_json(dataset/'metadata.json',out);write_json(dataset/'dataset.json',report)
 return report

def prepare(dataset,config=Config()):
 dataset=Path(dataset);decl=json.loads((dataset/'dataset.json').read_text())
 if not decl.get('research_pilot_ready'):raise ValueError('Research corpus not ready')
 assert decl['protocol']=='automatic_conditional_multivoice_pilot_v0.2' and decl['human_reviewed'] is False
 rows=json.loads((dataset/'metadata.json').read_text())
 code_names=['mandia_voice.py','corpus.py','research_corpus.py','train_voice.py']
 feature_config={k:asdict(config)[k] for k in ['mel','sample_rate','n_fft','hop']}
 fingerprint=hashlib.sha256(json.dumps({'declaration':decl,'metadata_sha256':sha(dataset/'metadata.json'),
  'code':{n:sha(Path(__file__).parent/n) for n in code_names},'config':feature_config},sort_keys=True).encode()).hexdigest()
 cache=dataset/'features'/fingerprint;cache.mkdir(parents=True,exist_ok=True)
 if (cache/'source.json').exists():
  prior=json.loads((cache/'source.json').read_text());assert prior['fingerprint']==fingerprint
  return cache
 total,square,frames=torch.zeros(config.mel),torch.zeros(config.mel),0
 for i,row in enumerate(rows):
  audio=Path(row['audio_absolute']);assert sha(audio)==row['audio_sha256']
  wave=torch.from_numpy(read_wave(audio,config));mel=to_mel(wave,config)
  if row['split']=='train':total+=mel.sum(0);square+=mel.square().sum(0);frames+=len(mel)
  feature=cache/f'{i:06d}.pt';temp=feature.with_suffix('.tmp')
  torch.save({'wave':wave,'mel':mel,'tokens':torch.tensor(tokenize(row['text'])),'speaker_index':row['speaker_index']},temp);temp.replace(feature)
  row['feature']=str(feature.relative_to(dataset));row['feature_sha256']=sha(feature)
  if (i+1)%50==0:print('FEATURES',i+1,'/',len(rows),flush=True)
 mean=total/frames;std=(square/frames-mean.square()).clamp_min(1e-4).sqrt()
 write_json(cache/'normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'train_frames':frames})
 write_json(cache/'rows.json',rows)
 source={'pilot_ready':True,'research_pilot_ready':True,'audit_protocol':'automatic_conditional_multivoice_pilot_v0.2',
  'human_validated_corpus':False,'release_ready':False,'feature_config':feature_config,'fingerprint':fingerprint,
  'counts':decl['counts'],'seconds':decl['seconds'],'split_protocol':decl['split_protocol']}
 write_json(cache/'source.json',source)
 return cache
