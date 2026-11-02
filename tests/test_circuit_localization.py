import random
import unittest

import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from continual.circuit_localization import intervene,make_pair,observations


class CircuitLocalizationTests(unittest.TestCase):
    def test_counterfactual_probe_answers_follow_actual_records_and_strings(self):
        import json,re
        rng=random.Random(42)
        for family in ['lookup','equality','copy','constant']:
            for i in range(30):
                pair=make_pair(rng,family,str(i))
                self.assertEqual({pair['clean']['answer'],pair['corrupt']['answer']},{0,1})
                self.assertNotEqual(pair['clean']['prompt'],pair['corrupt']['prompt'])
                if family=='lookup':
                    for side in ['clean','corrupt']:
                        text=pair[side]['prompt'];rows=json.loads(re.search(r'\[.*\]',text).group())
                        quoted=re.findall(r'"([xy])"',text)
                        # The query occurs after the displayed record array in every template.
                        key=quoted[-1]
                        if text.startswith('Find'):
                            key=re.search(r'="([xy])"',text).group(1)
                        row=next(r for r in rows if key in r.values())
                        self.assertEqual(next(v for v in row.values() if isinstance(v,int)),pair[side]['answer'])

    def test_intervention_is_position_and_channel_specific_and_cleans_up(self):
        torch.manual_seed(4)
        config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
                           num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        model=Qwen3ForCausalLM(config).eval();ids=torch.tensor([[2,3,4,5],[6,7,8,9]])
        with torch.no_grad(),observations(model) as cache:
            original=model(ids,use_cache=False).logits
        donor=cache[('z',0)].clone();donor[:,0]+=100
        with torch.no_grad(),intervene(model,'z',0,donor,[0]):
            altered=model(ids,use_cache=False).logits
        torch.testing.assert_close(altered[:,:-1],original[:,:-1],rtol=0,atol=0)
        self.assertGreater((altered[:,-1]-original[:,-1]).abs().max().item(),0.001)
        with torch.no_grad():torch.testing.assert_close(model(ids,use_cache=False).logits,original)
        model.requires_grad_(False);model.enable_input_require_grads()
        with observations(model,grad=True) as states:
            out=model(ids,use_cache=False)
            out.logits[:,-1,3].sum().backward()
        self.assertIsNotNone(states[('z',0)].grad)
        self.assertIsNotNone(states[('head',0)].grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
