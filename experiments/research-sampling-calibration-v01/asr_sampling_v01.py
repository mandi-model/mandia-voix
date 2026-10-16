from pathlib import Path
import sys,json,collections,argparse
P=Path('/content/drive/MyDrive/IvoireTTS_v01/06_mandia_preparation_v1');assert (P.parent/'pipeline_state.json').is_file();sys.path.insert(0,str(P/'code'))
from prepare_ivoiretts_gpu_v9r6 import ASR,resample,sha,jsave
from audit_spoken_forms_v01 import spoken_form
from rapidfuzz.distance import Levenshtein
import soundfile as sf
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['development','fresh'],required=True);stage=parser.parse_args().stage;D=P/'sampling-calibration-v01';E=D/stage;audio=json.loads((E/'AUDIO-RECEIPTS.json').read_text());asr=ASR().get('turbo');details=[];groups={}
for row in audio['items']:
 if row['kind']!='raw':continue
 file=Path(row['file']);assert sha(file)==row['sha256'];wave,sr=sf.read(file,dtype='float32');segments,_=asr.transcribe(resample(wave,sr,16000),language='fr',beam_size=5,condition_on_previous_text=False,vad_filter=False,temperature=0);recognized=' '.join(s.text.strip() for s in segments).strip();a=spoken_form(row['text']);b=spoken_form(recognized);we=Levenshtein.distance(a.split(),b.split());ce=Levenshtein.distance(a,b);tokens=b.split();repeated=sum(max(0,v-1) for v in collections.Counter(tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))).values());g=groups.setdefault(row['condition'],{'words':0,'word_edits':0,'characters':0,'character_edits':0,'repeated_trigram_excess':0,'temperature':row['temperature']});g['words']+=len(a.split());g['word_edits']+=we;g['characters']+=len(a);g['character_edits']+=ce;g['repeated_trigram_excess']+=repeated;details.append(dict(row,recognized=recognized,spoken_WER=we/max(1,len(a.split()))));print('SAMPLING_ASR',stage,row['condition'],row['id'],row['seed_index'],we,recognized,flush=True)
for g in groups.values():g.update(WER=g['word_edits']/g['words'],CER=g['character_edits']/g['characters'])
jsave(E/'ASR-COMPARISON.json',{'summary':groups,'details':details,'stage':stage,'scope':'8distinct texts times2seeds, not16 independent texts. Whisper turbo forcedfr beam5 temp0 noVAD; related to corpus screening. Spoken number normalization declared before evaluation.','human_naturalness_validated':False,'human_Ivorian_accent_validated':False})
if stage=='development':
 baseline=groups['t1000'];eligible=[(k,v) for k,v in groups.items() if k!='t1000' and v['word_edits']<=baseline['word_edits']-2 and v['character_edits']<=baseline['character_edits'] and v['repeated_trigram_excess']<=baseline['repeated_trigram_excess']];chosen=min(eligible,key=lambda kv:(kv[1]['word_edits'],kv[1]['character_edits'],abs(1-kv[1]['temperature']))) if eligible else ('t1000',baseline);jsave(D/'SELECTED-SAMPLING.json',{'condition':chosen[0],'temperature':chosen[1]['temperature'],'development_summary':groups,'fresh_used_for_selection':False,'no_model_weights_modified':True,'human_naturalness_validated':False})
print('SAMPLING_ASR_COMPLETE',stage,json.dumps(groups),flush=True)
