import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from continual.circuit_growth import GatedBank,attach
from continual.nonlinear_layers import backbone_digest


class GatedGrowthTests(unittest.TestCase):
    def test_parallel_bank_equals_widening_old_gated_projections(self):
        torch.manual_seed(4)
        old=GatedBank(8,12);new=GatedBank(8,3)
        with torch.no_grad():old.down.weight.normal_();new.down.weight.normal_()
        x=torch.randn(2,5,8)
        gate=torch.cat([old.gate.weight,new.gate.weight],dim=0)
        up=torch.cat([old.up.weight,new.up.weight],dim=0)
        down=torch.cat([old.down.weight,new.down.weight],dim=1)
        expected=torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x,gate))*torch.nn.functional.linear(x,up),down)
        torch.testing.assert_close(old(x)+new(x),expected)

    def test_only_added_channels_learn_and_bypass_restores_backbone(self):
        torch.manual_seed(7)
        cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        model=Qwen3ForCausalLM(cfg).eval();ids=torch.tensor([[3,4,5,6]])
        original=model(ids,use_cache=False).logits.detach();digest=backbone_digest(model)
        banks=attach(model,[0,1],4)
        torch.testing.assert_close(model(ids,use_cache=False).logits,original,atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in banks.parameters()),2*3*16*4)
        opt=torch.optim.AdamW(banks.parameters(),lr=0.1)
        model(ids,labels=ids,use_cache=False).loss.backward();opt.step()
        self.assertEqual(backbone_digest(model),digest)
        self.assertGreater((model(ids,use_cache=False).logits-original).abs().max().item(),0.001)
        model.temporary_bypass={0,1}
        torch.testing.assert_close(model(ids,use_cache=False).logits,original,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
