import copy
import unittest

import torch
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

from continual.causal_diagnosis import answer_logp, capture, greedy, patch, replace_output, reverted


class TinyTokenizer:
    eos_token_id = 31
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]['content']

    def __call__(self, texts, **kwargs):
        ids = [[int(x) for x in text.split()] for text in texts]
        width = max(map(len, ids))
        return BatchEncoding({'input_ids':torch.tensor([[0]*(width-len(x))+x for x in ids]),
                              'attention_mask':torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in ids])})

    def batch_decode(self, rows, **kwargs):
        return [' '.join(str(x) for x in row if x not in [0,31]) for row in rows]


class CausalDiagnosisTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8)
        self.old = Qwen3ForCausalLM(config).eval()
        self.new = copy.deepcopy(self.old)
        with torch.no_grad():
            self.new.model.layers[0].mlp.down_proj.weight.add_(torch.randn_like(
                self.new.model.layers[0].mlp.down_proj.weight)*0.5)
        self.inputs = {'input_ids': torch.tensor([[3,4,5,6], [7,8,9,10]]),
                       'attention_mask': torch.ones(2,4,dtype=torch.long)}

    def test_shifted_scoring_ignores_prompt_and_padding(self):
        logits = torch.zeros(1,5,3)
        logits[0,1,2] = 5
        logits[0,2,1] = 4
        labels = torch.tensor([[-100,-100,2,1,-100]])
        expected = (logits[0,1].log_softmax(0)[2]+logits[0,2].log_softmax(0)[1])/2
        torch.testing.assert_close(answer_logp(logits,labels)[0], expected)
        logits[0,0] = 999; logits[0,3] = -999
        torch.testing.assert_close(answer_logp(logits,labels)[0], expected)

    @torch.inference_mode()
    def test_known_causal_module_and_controls(self):
        sites = ['0:mlp','0:self_attn','1:residual']
        old_output, cache = capture(self.old,self.inputs,sites)
        new_output = self.new(**self.inputs,use_cache=False)
        self.assertGreater((old_output.logits-new_output.logits).abs().max().item(),0.001)
        with patch(self.new,'0:mlp',cache['0:mlp']):
            restored = self.new(**self.inputs,use_cache=False)
        torch.testing.assert_close(restored.logits,old_output.logits)
        # The unchanged attention module cannot repair the changed MLP.
        with patch(self.new,'0:self_attn',cache['0:self_attn']):
            control = self.new(**self.inputs,use_cache=False)
        torch.testing.assert_close(control.logits,new_output.logits)
        with patch(self.new,'1:residual',cache['1:residual']):
            copied = self.new(**self.inputs,use_cache=False)
        torch.testing.assert_close(copied.logits,old_output.logits)

    @torch.inference_mode()
    def test_weight_restore_even_after_error(self):
        expected_new = self.new(**self.inputs,use_cache=False).logits.clone()
        expected_old = self.old(**self.inputs,use_cache=False).logits.clone()
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with reverted(self.new,self.old,['0:mlp']):
                torch.testing.assert_close(self.new(**self.inputs,use_cache=False).logits,expected_old)
                raise RuntimeError('intentional')
        torch.testing.assert_close(self.new(**self.inputs,use_cache=False).logits,expected_new)
        torch.testing.assert_close(self.old(**self.inputs,use_cache=False).logits,expected_old)
        with self.assertRaises(ValueError):
            with reverted(self.new,self.old,['0:residual']):
                pass

    def test_tuple_preservation_and_shape_rejection(self):
        marker = object()
        output = (torch.zeros(2,3),marker)
        changed = replace_output(output,torch.ones(2,3))
        self.assertIs(changed[1],marker)
        with self.assertRaises(ValueError):
            replace_output(output,torch.ones(3,2))

    @torch.inference_mode()
    def test_cached_generation_matches_hf_and_self_and_output_copy(self):
        tokenizer = TinyTokenizer()
        items = [{'id':str(i),'prompt':text,'answer':[],'skill':'C'}
                 for i,text in enumerate(['3 4','7 8 9'])]
        generated = greedy(self.new,tokenizer,items,maximum=6)
        inputs = tokenizer([x['prompt'] for x in items])
        expected = self.new.generate(**inputs,max_new_tokens=6,do_sample=False,
                                     pad_token_id=0,eos_token_id=31)
        self.assertEqual([x['response'] for x in generated],tokenizer.batch_decode(expected[:,3:].tolist()))
        self_patch = greedy(self.new,tokenizer,items,maximum=6,donor=self.new,site='0:mlp')
        self.assertEqual([x['response'] for x in generated],[x['response'] for x in self_patch])
        before = greedy(self.old,tokenizer,items,maximum=6)
        copied = greedy(self.new,tokenizer,items,maximum=6,donor=self.old,site='1:residual')
        self.assertEqual([x['response'] for x in before],[x['response'] for x in copied])


if __name__ == '__main__':
    unittest.main()
