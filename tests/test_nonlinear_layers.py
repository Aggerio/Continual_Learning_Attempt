import copy
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from continual.nonlinear_layers import attach, backbone_digest, capture_states, normalized_state_loss


class NonlinearLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
                           num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        self.model=Qwen3ForCausalLM(config).eval()
        self.inputs=torch.tensor([[3,4,5,6],[7,8,9,10]])

    def test_zero_initialized_expansion_is_identity_and_only_new_weights_learn(self):
        baseline=self.model(self.inputs,use_cache=False).logits.detach()
        digest=backbone_digest(self.model)
        extra=attach(self.model,[0,1],4)
        torch.testing.assert_close(self.model(self.inputs,use_cache=False).logits,baseline,rtol=0,atol=0)
        optimizer=torch.optim.AdamW(extra.parameters(),lr=0.1)
        output=self.model(self.inputs,labels=self.inputs,use_cache=False)
        output.loss.backward()
        self.assertGreater(extra['0'].down.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for n,p in self.model.named_parameters() if not n.startswith('temporary_layers.')))
        optimizer.step()
        self.assertEqual(digest,backbone_digest(self.model))
        self.assertGreater((self.model(self.inputs,use_cache=False).logits-baseline).abs().max().item(),0.001)
        self.model.temporary_bypass={0,1}
        torch.testing.assert_close(self.model(self.inputs,use_cache=False).logits,baseline,rtol=0,atol=0)

    def test_hint_gradient_and_capture_after_temporary_layer(self):
        extra=attach(self.model,[0],4)
        with torch.no_grad():extra['0'].down.weight.normal_(std=0.1)
        with capture_states(self.model,[0]) as states:
            self.model(self.inputs,use_cache=False)
        target=torch.randn_like(states[0])
        loss=normalized_state_loss(states[0],target)
        loss.backward()
        self.assertGreater(extra['0'].down.weight.grad.abs().sum().item(),0)
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(normalized_state_loss(states[0].detach(),states[0].detach()*2),torch.tensor(0.),atol=1e-6,rtol=0)

    def test_serialized_layers_reload_without_changing_base(self):
        other=copy.deepcopy(self.model)
        extra=attach(self.model,[0,1],4)
        with torch.no_grad():extra['1'].down.weight.normal_(std=0.1)
        other_extra=attach(other,[0,1],4)
        other_extra.load_state_dict(extra.state_dict())
        torch.testing.assert_close(self.model(self.inputs,use_cache=False).logits,other(self.inputs,use_cache=False).logits)
        self.assertEqual(sum(p.numel() for p in extra.parameters()),2*(16+16*4*2))
        with self.assertRaises(ValueError):attach(other,[0],4)


if __name__=='__main__':unittest.main()
