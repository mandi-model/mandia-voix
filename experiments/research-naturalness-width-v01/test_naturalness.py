import unittest,tempfile,torch
from pathlib import Path
from mandia_voice import Config
from phase_vocoder import ConsistentVocoder
from natural_vocoder import widen,NaturalDiscriminator,natural_generator_loss
import checkpoint_store as store

class NaturalnessTests(unittest.TestCase):
    def test_widen_preserves_waveform_but_can_break_symmetry(self):
        torch.manual_seed(23);torch.set_num_threads(2);base=ConsistentVocoder(Config(vocoder_dim=16,vocoder_layers=2));wide=widen(base);mel=torch.randn(1,300,80)
        with torch.no_grad():torch.testing.assert_close(wide(mel),base(mel),rtol=3e-5,atol=3e-5)
        opt=torch.optim.AdamW(wide.parameters(),lr=.001);loss=wide(mel).square().mean();loss.backward();opt.step()
        self.assertGreater(float((wide.input.weight[:16]-wide.input.weight[16:]).detach().abs().sum()),0)
    def test_multiscale_waveform_adversary_has_finite_generator_gradients(self):
        torch.set_num_threads(2);disc=NaturalDiscriminator().requires_grad_(False);fake=(torch.randn(1,24576)*.03).requires_grad_();real=torch.randn_like(fake)*.03
        loss=natural_generator_loss(fake,real,disc,Config(),fake.new_tensor(.1));loss.backward();self.assertTrue(torch.isfinite(fake.grad).all());self.assertGreater(float(fake.grad.abs().sum()),0)
    def test_immutable_snapshot_preserves_optimizer_rng_and_older_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters());model(torch.ones(1,2)).sum().backward();opt.step();torch.manual_seed(900)
            state={'step':1,'signature':'test','validation':2.,'model':model.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state()}
            row=store.save(root,state);digest=store.sha(root/row['file']);expected=torch.rand(4);state2=dict(state,step=2,validation=1.);store.save(root,state2);loaded_row,loaded=store.latest(root,'test');self.assertEqual(loaded_row['step'],2);self.assertEqual(store.sha(root/row['file']),digest)
            torch.set_rng_state(loaded['torch_rng']);torch.testing.assert_close(torch.rand(4),expected);self.assertTrue(loaded['optimizer']['state'])

if __name__=='__main__':unittest.main()
