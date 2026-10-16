"""Colab-only audited SIWIS + frozen IvoireTTS features for a phonetic-input experiment."""
from pathlib import Path
import sys,json,hashlib,io,time,collections,unicodedata,csv
P=Path('/content/drive/MyDrive/IvoireTTS_v01/06_mandia_preparation_v1');assert (P.parent/'pipeline_state.json').is_file()
C=P/'code/research-phoneme-v01';sys.path.insert(0,str(C))
import torch,numpy as np,soundfile as sf,pyarrow.parquet as pq
from mandia_voice import Config,to_mel,tokenize,canonical_text
from phoneme_frontend import version,phonemes
from corpus import sha,write_json
D=P/'pronunciation-phoneme-v01';D.mkdir(exist_ok=True);F=D/'features';F.mkdir(exist_ok=True)
config=Config();torch.set_num_threads(2)
manifest=json.loads((P/'code/SIWIS-MIRROR-MANIFEST.json').read_text());raw=P/'external-corpora/siwis-mirror-24k-v03'
assert json.loads((raw/'STATE.json').read_text())['phase']=='all_shards_verified'
source_signature={'mirror':manifest,'frontend':version(),'config':{k:getattr(config,k) for k in ['mel','sample_rate','n_fft','hop']},'preparation_sha256':sha(Path(__file__)),'frontend_sha256':sha(C/'phoneme_frontend.py'),'model_sha256':sha(C/'mandia_voice.py')}
if (F/'READY.json').exists():
 assert json.loads((F/'READY.json').read_text())['signature']==source_signature
 print('PHONEME_FEATURES_ALREADY_VERIFIED',flush=True);sys.exit(0)
# Metadata pass first; keep whole sentence/source groups together, including re-readings.
records=[]
for f in sorted((raw/'data').glob('*.parquet')):
 for batch in pq.ParquetFile(f).iter_batches(batch_size=128,columns=['clip_id','transcript','transcript_raw','source_file','part','style','session','duration_s','sample_rate']):
  records+=batch.to_pylist()
assert len(records)==9750 and len({r["clip_id"] for r in records})==9750
reserved=list(csv.DictReader((P/'code/research-duration-control-v01/phrases-test-reservees.csv').open()))
fresh=[{'id':'p-new-001','text':"Bonjour, je voudrais envoyer ce colis à Bouaké demain matin."},{'id':'p-new-002','text':"Nous avons rendez-vous à quinze heures trente devant la pharmacie."},{'id':'p-new-003','text':"Le prix est de deux mille cinq cents francs, livraison comprise."},{'id':'p-new-004','text':"Peux-tu fermer doucement la porte et laisser la fenêtre ouverte ?"},{'id':'p-new-005','text':"Il a plu cette nuit, mais la route est maintenant praticable."},{'id':'p-new-006','text':"Avant de partir pour Yamoussoukro, vérifie ton billet et appelle ta sœur."}]
write_json(D/'fresh-test-texts.json',fresh)
blocked={canonical_text(r['text']) for r in reserved+fresh}
# Union by normalized prompt and original sentence grouping, avoiding emphasis leakage.
parent={}
def find(x):
 parent.setdefault(x,x)
 while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
 return x
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:parent[max(a,b)]=min(a,b)
for r in records:union('text:'+canonical_text(r['transcript']),'source:'+r['source_file'])
groups={r['clip_id']:find('text:'+canonical_text(r['transcript'])) for r in records}
rows=[];rejected=collections.Counter();audio_seen={};group_splits={};payload=[];shard_index=0
stats_total=torch.zeros(config.mel,dtype=torch.float64);stats_square=stats_total.clone();stats_frames=0
# Stable shard writing; interrupted attempts are retained as same-content rebuilds.
def flush():
 global shard_index,payload
 if not payload:return
 name=f'{shard_index:04d}.pt';target=F/name;temp=F/(name+'.tmp');torch.save(payload,temp);temp.replace(target);digest=sha(target)
 for item in rows[-len(payload):]:item['feature']=str(target);item['feature_sha256']=digest
 shard_index+=1;payload=[]
 write_json(F/'PROGRESS.json',{'phase':'preparing','accepted':len(rows),'rejected':dict(rejected),'shards':shard_index})
 print('PHONEME_FEATURES',len(rows),'shards',shard_index,flush=True)
def append(sample,row):
 global stats_total,stats_square,stats_frames
 if row['split']=='train' and row['domain']=='siwis':
  mel=sample['mel'].double();stats_total+=mel.sum(0);stats_square+=mel.square().sum(0);stats_frames+=len(mel)
 row['feature_index']=len(payload);rows.append(row);payload.append(sample)
 if len(payload)>=128:flush()
for f in sorted((raw/'data').glob('*.parquet')):
 for batch in pq.ParquetFile(f).iter_batches(batch_size=32):
  for r in batch.to_pylist():
   text=r['transcript'];key=canonical_text(text);group=groups[r['clip_id']]
   if key in blocked:rejected['reserved_text']+=1;continue
   # Whole prompt groups are held out deterministically. This is text-group, not session, generalization.
   bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
   split='validation' if bucket==0 else 'test' if bucket==1 else 'train'
   raw_audio=r['audio']['bytes'];digest=hashlib.sha256(raw_audio).hexdigest()
   if digest in audio_seen:
    if audio_seen[digest]!=key:raise ValueError('Identical audio has inconsistent prompts')
    rejected['duplicate_audio']+=1;continue
   wave,sr=sf.read(io.BytesIO(raw_audio),dtype='float32');assert sr==24000 and wave.ndim==1
   seconds=len(wave)/sr;peak=float(np.max(np.abs(wave)));rms=float(np.sqrt(np.mean(wave**2)))
   if not np.isfinite(wave).all() or not .9<=seconds<=13 or rms<.002 or float((np.abs(wave)>=.999).mean())>.005:rejected['signal_or_duration']+=1;continue
   tokens=tokenize(text);mel=to_mel(torch.from_numpy(wave),config)
   if len(tokens)>len(mel) or len(tokens)>700:rejected['alignment_length']+=1;continue
   audio_seen[digest]=key;group_splits.setdefault(group,set()).add(split)
   row={'id':r['clip_id'],'text':text,'split':split,'domain':'siwis','group':group,'speaker_index':0,'audio_sha256':digest,'seconds':seconds,'frontend':phonemes(text),'part':r['part'],'session':r['session']}
   append({'wave':torch.from_numpy(wave),'mel':mel.half(),'tokens':torch.tensor(tokens),'speaker_index':0},row)
flush();assert stats_frames>0
mean=stats_total/stats_frames;std=(stats_square/stats_frames-mean.square()).clamp_min(1e-4).sqrt()
write_json(F/'normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'train_frames':stats_frames,'fit_on':'SIWIS training only; frozen unchanged for Ivoire adaptation'})
# Re-tokenize the existing immutable IvoireTTS corpus, preserving every original split.
old=Path(json.loads((P/'joint-clean-v01/STATUS.json').read_text())['features']);ivoire=json.loads((old/'rows.json').read_text())
for r in ivoire:
 if canonical_text(r['text']) in blocked:rejected['ivoire_reserved']+=1;continue
 file=P/r['feature'] if not Path(r['feature']).is_absolute() else Path(r['feature'])
 # Feature paths in the frozen corpus are relative to its run root.
 if not file.is_file():
  alternatives=[P/'word-clean-v01'/r['feature'],P/'word-clean-v01'/Path(r['feature']).name,old/Path(r['feature']).name]
  file=next((x for x in alternatives if x.is_file()),file)
 assert sha(file)==r['feature_sha256']
 old_sample=torch.load(file,map_location='cpu',weights_only=True);wave=old_sample['wave'].float();mel=old_sample['mel'].float();tokens=tokenize(r['text'])
 if len(tokens)>len(mel) or len(tokens)>700:rejected['ivoire_alignment_length']+=1;continue
 row={'id':r['id'],'text':r['text'],'split':r['split'],'domain':'ivoire','group':r['source_id'],'speaker_index':1,'audio_sha256':r.get('audio_sha256'),'seconds':len(wave)/24000,'frontend':phonemes(r['text']),'original_feature_sha256':r['feature_sha256']}
 append({'wave':wave,'mel':mel.half(),'tokens':torch.tensor(tokens),'speaker_index':1},row)
flush();assert all(len(s)==1 for s in group_splits.values())
counts=dict(collections.Counter((r['domain']+'/'+r['split']) for r in rows));durations={k:sum(r['seconds'] for r in rows if r['domain']+'/'+r['split']==k) for k in counts}
assert counts['siwis/train']>8000 and counts['siwis/validation']>=20 and counts['siwis/test']>=20
assert counts['ivoire/train']>=900
write_json(F/'rows.json',rows);write_json(F/'source.json',{'pilot_ready':True,'counts':counts,'seconds':durations,'rejected':dict(rejected),'SIWIS_split':'normalized text + source_file union groups, SHA256 98/1/1; no claim of session-disjoint SIWIS testing','Ivoire_split':'preserved frozen source-disjoint 980/8/12, modulo declared signal/text-length exclusions','human_transcripts_checked':False,'SIWIS_source':manifest,'accent_confirmed_for_one_real_Ivoire_reference_only':True,'feature_config':source_signature['config'],'normalization':'SIWIS train only, frozen for adaptation','frontend':source_signature['frontend']})
write_json(F/'READY.json',{'signature':source_signature,'rows_sha256':sha(F/'rows.json'),'normalization_sha256':sha(F/'normalization.json'),'counts':counts,'seconds':durations})
print('PHONEME_CORPUS_READY',json.dumps({'counts':counts,'hours':{k:v/3600 for k,v in durations.items()}}),flush=True)
