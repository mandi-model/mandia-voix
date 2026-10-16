"""eSpeak NG French phonetic transcription only. No synthesized audio is used."""
from functools import lru_cache
import subprocess,unicodedata,re
@lru_cache(maxsize=30000)
def phonemes(text):
 text=unicodedata.normalize('NFC',text).replace('’',"'").strip()
 if not text:raise ValueError('Empty text')
 # One token is an eSpeak ASCII phonetic character, including its stress/length marks.
 # -x is phonetic transcription, -q disables speech output. Keep case: S and s differ.
 result=subprocess.run(['espeak-ng','-q','-v','fr-fr','-x','--stdin'],input=text,text=True,capture_output=True,check=True).stdout
 result=' '.join(result.split())
 if not result or any(ord(c)<32 or ord(c)>126 for c in result):raise ValueError('Invalid ASCII phonetic transcription')
 return result
def tokenize(text):
 return [1]+[ord(c)+2 for c in phonemes(text)]+[1]
def version():return subprocess.run(['espeak-ng','--version'],text=True,capture_output=True,check=True).stdout.strip()
