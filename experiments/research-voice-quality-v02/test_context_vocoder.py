import random,unittest,torch
from mandia_voice import Config,FourierVocoder
from context_vocoder import CORE,CONTEXT,crop,center_wave,receptive_field,QualityDiscriminator,generator_loss

class ContextTests(unittest.TestCase):
    def test_scored_center_agrees_with_full_utterance(self):
        torch.manual_seed(33);torch.set_num_threads(2)
        model=FourierVocoder(Config()).eval();mel=torch.randn(1,430,80)
        with torch.no_grad():
            full=model(mel);offset=45
            partial=center_wave(model,mel[:,offset:offset+CORE+2*CONTEXT+1])
        begin=(offset+CONTEXT)*256
        self.assertEqual(partial.shape,(1,CORE*256))
        torch.testing.assert_close(partial,full[:,begin:begin+CORE*256],rtol=1e-4,atol=2e-5)
        self.assertEqual(receptive_field(model),181)
        self.assertGreater(CONTEXT,(receptive_field(model)-1)//2+2)
    def test_actual_crop_sample_alignment_and_no_padding(self):
        total=CORE+2*CONTEXT
        sample={'mel':torch.zeros(total+1,80),'wave':torch.arange(total*256).float()}
        mel,wave=crop([sample],Config(),random.Random(1))
        torch.testing.assert_close(wave[0],sample['wave'][CONTEXT*256:(CONTEXT+CORE)*256])
        with self.assertRaises(ValueError):crop([{'mel':sample['mel'][:-1],'wave':sample['wave']}],Config(),random.Random(1))
    def test_generator_gets_finite_periodic_discriminator_gradients(self):
        torch.set_num_threads(2);disc=QualityDiscriminator().requires_grad_(False)
        fake=(.01*torch.randn(1,CORE*256)).requires_grad_();real=.01*torch.randn_like(fake)
        loss=generator_loss(fake,real,disc,Config());loss.backward()
        self.assertTrue(torch.isfinite(fake.grad).all());self.assertGreater(float(fake.grad.abs().sum()),0)

if __name__=='__main__':unittest.main()
