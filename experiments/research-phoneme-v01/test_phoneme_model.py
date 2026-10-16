import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np,torch
from fast_alignment import maximum_path,beta_binomial_prior
from mandia_voice import Config,Acoustic,FourierVocoder,parameter_counts
class AlignmentTests(unittest.TestCase):
 def test_path_is_surjective_monotonic_and_optimal(self):
  import itertools
  rng=np.random.default_rng(42)
  for frames,tokens in [(4,2),(6,3),(8,4)]:
   scores=rng.normal(size=(frames,tokens)).astype('float32');path=maximum_path(scores)
   candidates=[]
   for cuts in itertools.combinations(range(1,frames),tokens-1):
    candidate=np.repeat(np.arange(tokens),np.diff([0,*cuts,frames]));candidates.append(float(scores[np.arange(frames),candidate].sum()))
   self.assertAlmostEqual(float(scores[np.arange(frames),path].sum()),max(candidates),places=5)
   self.assertEqual(set(path.tolist()),set(range(tokens)));self.assertTrue(np.isin(np.diff(path),[0,1]).all())
 def test_prior_finite_ordered(self):
  a=beta_binomial_prior(80,12);self.assertEqual(a.shape,(80,12));self.assertTrue(np.isfinite(a).all());self.assertLess(a[:5].argmax(-1).mean(),a[-5:].argmax(-1).mean())
 def test_real_french_frontend_if_available(self):
  import shutil
  if not shutil.which('espeak-ng'):self.skipTest('Colab checks the installed eSpeak backend')
  from phoneme_frontend import phonemes,tokenize
  text="Les enfants ont vingt-trois livres et deux cents francs."
  phones=phonemes(text);self.assertTrue(phones.strip());self.assertTrue(all(32<=ord(x)<=126 for x in phones))
  self.assertEqual(tokenize(text),[1]+[ord(c)+2 for c in phones]+[1])
 def test_end_to_end_gradients_and_padding(self):
  torch.set_num_threads(2);torch.manual_seed(42)
  c=Config(dim=32,encoder_layers=2,flow_layers=2,vocoder_dim=16,vocoder_layers=2)
  m=Acoustic(c);tokens=torch.randint(2,90,(2,12));lengths=torch.tensor([12,8]);tokens[1,8:]=0
  mel=torch.randn(2,220,80);nf=torch.tensor([220,160]);speaker=torch.tensor([0,1]);m.training_step=0
  parts=m.losses(tokens,lengths,mel,nf,speaker_ids=speaker,generator=torch.Generator().manual_seed(1));sum(parts.values()).backward()
  self.assertTrue(all(torch.isfinite(x) for x in parts.values()));self.assertGreater(float(m.embedding.weight.grad.abs().sum()),0)
  self.assertGreater(float(m.duration[0].weight.grad.abs().sum()),0);self.assertGreater(float(m.flow_out.weight.grad.abs().sum()),0)
  m.eval();wave,n=m.synthesize(tokens,lengths,speaker_ids=speaker,steps=2)
  self.assertTrue(torch.isfinite(wave).all());self.assertEqual(float(wave[1,int(n[1]):].abs().sum()),0)
if __name__=='__main__':
 print('parameter_counts',parameter_counts(Config()))
 unittest.main()
