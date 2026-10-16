"""Download public inference weights and verify SHA256."""
import argparse,hashlib,json,os,urllib.request
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",default="weights");a=p.parse_args()
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((Path(__file__).parent/"WEIGHTS.json").read_text())
 for item in manifest["files"]:
  path=root/item["name"];tmp=path.with_suffix(".download")
  if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]:continue
  h=hashlib.sha256()
  with urllib.request.urlopen(item["url"]) as src,tmp.open("wb") as dst:
   for chunk in iter(lambda:src.read(1024*1024),b""):h.update(chunk);dst.write(chunk)
  if h.hexdigest()!=item["sha256"]:
   tmp.unlink();raise ValueError("SHA256 mismatch: "+item["name"])
  os.replace(tmp,path);print("Verified",path)
if __name__=="__main__":main()

