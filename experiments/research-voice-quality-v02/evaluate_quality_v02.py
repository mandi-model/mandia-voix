"""Paired development evaluation: same acoustic checkpoint, text, ODE and seed.
The six texts were already inspected in v01; they are development probes here.
"""
from pathlib import Path
import json,sys,time
P=Path('/content/drive/MyDrive/IvoireTTS_v01/06_mandia_preparation_v1');assert (P.parent/'pipeline_state.json').is_file()
sys.path.insert(0,str(P/'code/research-phoneme-v01'))
import torch,soundfile as sf
from mandia_voice import Config,Acoustic,FourierVocoder,tokenize,spectral_loss
from corpus import sha,write_json
torch.set_num_threads(2);device=torch.device('cuda');config=Config()
D=P/'pronunciation-phoneme-v01';F=D/'features';R=P/'voice-quality-v02';E=R/'development-evaluation';E.mkdir(exist_ok=True)
rows=json.loads((F/'rows.json').read_text());lookup={r['id']:r for r in rows}
old=json.loads((D/'evaluation/AUDIO-RECEIPTS.json').read_text())
norm=json.loads((F/'normalization.json').read_text());mean=torch.tensor(norm['mean'],device=device);std=torch.tensor(norm['std'],device=device)
def read(path):
    assert sha(path)==json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
    return torch.load(path,map_location=device,weights_only=True)
file=R/'vocoder/best.pt';state=read(file);voc=FourierVocoder(config).to(device).eval();voc.load_state_dict(state['model']);selected=state['step'];del state
acoustic_file=D/'siwis/best.pt';assert sha(acoustic_file)=='b65fe825341d27bf3a088b5079b29b9f24a6f410b584f15a91143ee73002e758'
model=Acoustic(config).to(device).eval();model.load_state_dict(read(acoustic_file)['model']);items=[];recon=[]
def output(wave,name,row,condition):
    assert torch.isfinite(wave).all();peak=float(wave.abs().max());scale=min(1.,.95/max(peak,1e-6));path=E/(name+'.wav');sf.write(path,wave.cpu().numpy()*scale,24000,subtype='PCM_16')
    return dict(id=row['id'],text=row['text'],condition=condition,file=str(path),sha256=sha(path),seconds=len(wave)/24000,peak_before_scale=peak,scale=scale)
with torch.inference_mode():
    for r in old['items']:
        if r['condition']=='siwis':
            assert sha(Path(r['file']))==r['sha256'];items.append(dict(r,condition='development/old_vocoder'))
        elif '/' in r['condition']:
            assert sha(Path(r['file']))==r['sha256'];items.append(dict(r,condition=r['condition'].replace('vocoder_real_mel','old_vocoder_real_mel')))
    fresh=json.loads((D/'fresh-test-texts.json').read_text())
    for i,r in enumerate(fresh):
        text=torch.tensor([tokenize(r['text'])],device=device);nt=torch.tensor([text.shape[1]],device=device)
        torch.cuda.synchronize();start=time.perf_counter();mel,n=model.synthesize(text,nt,steps=32,seed=900+i,speaker_ids=torch.zeros(1,dtype=torch.long,device=device));wave=voc(mel*std+mean)[0];torch.cuda.synchronize();elapsed=time.perf_counter()-start
        item=output(wave,'new-'+r['id'],r,'development/new_vocoder');item['rtf']=elapsed/item['seconds'];items.append(item)
    for r in old['items']:
        if not r['condition'].endswith('/real'):continue
        row=lookup[r['id']];assert sha(Path(row['feature']))==row['feature_sha256'];sample=torch.load(row['feature'],map_location='cpu',weights_only=True)[row['feature_index']]
        wave=voc(sample['mel'].float().to(device)[None])[0]
        item=output(wave,'new-control-'+r['condition'].split('/')[0]+'-'+r['id'],r,r['condition'].replace('/real','/new_vocoder_real_mel'));items.append(item)
        n=min(len(wave),len(sample['wave']));value=float(spectral_loss(wave[None,:n],sample['wave'][None,:n].to(device)))
        recon.append({'id':row['id'],'domain':row['domain'],'new_full_utterance_spectral_loss':value})
write_json(E/'AUDIO-RECEIPTS.json',{'items':items,'reconstruction':recon,'new_vocoder_sha256':sha(file),'new_vocoder_selected_step':selected,'unchanged_acoustic_sha256':sha(acoustic_file),'old_vocoder_sha256':old['fixed_vocoder_sha256'],'fresh_test':False,'method':'Previously inspected six v01 texts, fixed acoustic, ODE32, identical seeds900+i. Real-mel controls from the same v01 test items. Checkpoint selection used validation only. Human listening required; no ASR evidence of echo removal.','inference_parameters':9368355,'human_echo_resolved':False,'accent_validated':False})
print('QUALITY_DEVELOPMENT_AUDIO_SAVED',len(items),'selected_step',selected,flush=True)
