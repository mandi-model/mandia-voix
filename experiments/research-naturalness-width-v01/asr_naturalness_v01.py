from pathlib import Path
import sys,json
P=Path('/content/drive/MyDrive/IvoireTTS_v01/06_mandia_preparation_v1');assert (P.parent/'pipeline_state.json').is_file();sys.path.insert(0,str(P/'code'))
from prepare_ivoiretts_gpu_v9r6 import ASR,resample,normalized,sha,jsave
from audit_spoken_forms_v01 import spoken_form
from rapidfuzz.distance import Levenshtein
import soundfile as sf
E=P/'naturalness-width-v01/evaluation';receipt=json.loads((E/'AUDIO-RECEIPTS.json').read_text());asr=ASR().get('turbo');details=[];groups={}
for row in receipt['items']:
 if row['kind']=='listening':continue
 file=Path(row['file']);assert sha(file)==row['sha256'];wave,sr=sf.read(file,dtype='float32');segments,_=asr.transcribe(resample(wave,sr,16000),language='fr',beam_size=5,condition_on_previous_text=False,vad_filter=False,temperature=0)
 text=' '.join(s.text.strip() for s in segments).strip();item=dict(row,recognized=text)
 for metric,normalizer in [('raw',normalized),('spoken',spoken_form)]:
  a=normalizer(row['text']);b=normalizer(text);e=Levenshtein.distance(a.split(),b.split());nc=len(a);ec=Levenshtein.distance(a,b);n=len(a.split());value=groups.setdefault(row['condition'],{}).setdefault(metric,[0,0,0,0]);value[0]+=e;value[1]+=n;value[2]+=ec;value[3]+=nc;item[metric+'_WER']=e/max(n,1);item[metric+'_CER']=ec/max(nc,1)
 details.append(item);print('NATURALNESS_ASR',row['condition'],row['id'],item['spoken_WER'],text,flush=True)
summary={k:{metric:{'WER':v[0]/v[1],'CER':v[2]/v[3],'words':v[1]} for metric,v in values.items()} for k,values in groups.items()}
jsave(E/'ASR-COMPARISON.json',{'summary':summary,'details':details,'selected_checkpoints':receipt['selected'],'vocoder_latency':receipt['vocoder_latency'],'ASR':'Whisper large-v3-turbo, forced French beam5 temp0 noVAD no previous text; related to data screening','fresh_text_count':6,'limits':'One exploratory seed and six fresh texts. Prior development sentence reported separately. ASR does not measure robotics, naturalness, echo, accent or speaker similarity. Listening copies are level-matched and excluded from ASR.','human_naturalness_validated':False,'release_ready':False});print('NATURALNESS_ASR_REPORT_SAVED',json.dumps(summary),flush=True)
