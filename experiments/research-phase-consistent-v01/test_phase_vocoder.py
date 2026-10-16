import unittest,torch
from mandia_voice import Config,FourierVocoder
from phase_vocoder import ConsistentVocoder,project,relative_inconsistency

class PhaseTests(unittest.TestCase):
    def test_exact_initial_waveform_and_state_compatibility(self):
        torch.manual_seed(41);torch.set_num_threads(2);c=Config(vocoder_dim=16,vocoder_layers=2)
        old=FourierVocoder(c);new=ConsistentVocoder(c);new.load_state_dict(old.state_dict());mel=torch.randn(1,300,80)
        torch.testing.assert_close(new(mel),old(mel),rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in old.parameters()),sum(p.numel() for p in new.parameters()))
    def test_real_consistency_random_phase_and_finite_gradients(self):
        c=Config();window=torch.hann_window(c.n_fft);wave=torch.randn(1,40000)
        z=torch.stft(wave,c.n_fft,c.hop,window=window,return_complex=True);_,p=project(z,c,window)
        self.assertLess(float(relative_inconsistency(z,p)),1e-10)
        phase=torch.randn_like(z.real,requires_grad=True);bad=z.abs()*torch.complex(phase.cos(),phase.sin());_,r=project(bad,c,window);loss=relative_inconsistency(bad,r);loss.backward()
        self.assertGreater(float(loss.detach()),.01);self.assertTrue(torch.isfinite(phase.grad).all());self.assertGreater(float(phase.grad.abs().sum()),0)

if __name__=='__main__':unittest.main()
