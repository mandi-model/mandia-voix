"""Immutable per-step full states. No renaming/replacing rolling binary files."""
from pathlib import Path
import json,hashlib,uuid,torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(root,state,admitted=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if str(root).startswith('/content/drive/'):
        assert Path('/content/drive/MyDrive/IvoireTTS_v01/pipeline_state.json').is_file(),'Drive disconnected; no local fallback'
    name=f'step-{state["step"]:06d}-{uuid.uuid4().hex[:12]}.pt';file=root/name
    with file.open('xb') as stream:torch.save(state,stream)
    digest=sha(file);size=file.stat().st_size
    receipt={'file':name,'step':state['step'],'sha256':digest,'bytes':size,'validation':state['validation'],'signature':state['signature'],'admitted':bool(admitted)}
    metadata=file.with_suffix('.json')
    with metadata.open('x') as stream:json.dump(receipt,stream,indent=2)
    assert sha(file)==json.loads(metadata.read_text())['sha256']
    return receipt

def receipts(root,signature):
    out=[]
    for path in Path(root).glob('step-*.json'):
        try:
            row=json.loads(path.read_text())
            if row['signature']==signature:out.append(row)
        except (ValueError,KeyError):continue
    return out

def read(root,row,device='cpu'):
    root=Path(root);file=root/row['file'];assert file.parent==root and sha(file)==row['sha256'],'Snapshot hash mismatch'
    state=torch.load(file,map_location=device,weights_only=True);assert state['step']==row['step'] and state['signature']==row['signature']
    return state

def latest(root,signature):
    rows=sorted(receipts(root,signature),key=lambda r:(r['step'],r['file']),reverse=True)
    if not rows:return None,None
    # A corrupt newest snapshot stops explicitly; never silently rewind a run.
    row=rows[0];return row,read(root,row)

def best(root,signature):
    rows=[r for r in receipts(root,signature) if r['admitted']]
    if not rows:raise ValueError('No admitted checkpoint')
    row=min(rows,key=lambda r:(r['validation'],r['step']));return row,read(root,row)
