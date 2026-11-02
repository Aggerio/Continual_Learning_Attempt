import copy
import unittest
import torch
from peft import LoraConfig,get_peft_model
from transformers import Qwen3Config,Qwen3ForCausalLM
from continual.group_preservation import numeric_positions,numeric_logp,protect_adapters,module_prefix
from continual.tasks import compact


class CharacterTokenizer:
    eos_token='~'
    def encode(self,text,**kw):return [ord(c) for c in text]
    def __call__(self,text,**kw):return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def apply_chat_template(self,messages,**kw):return '123 prompt '+messages[-1]['content']+' => '


class GroupPreservationTests(unittest.TestCase):
    def test_numeric_alignment_excludes_prompt_and_keys_and_covers_every_digit(self):
        tok=CharacterTokenizer();item={'id':'x','skill':'D','prompt':'sum 12 + 3','answer':[{'code2':'x','total':15},{'code2':'y','total':28}]}
        p=numeric_positions(tok,item);text=tok.apply_chat_template([{'content':item['prompt']}])+compact(item['answer'])+'~'
        self.assertEqual(''.join(text[i+1] for i in p),'1528')
        item['skill']='A';item['answer']=[{'value2':19},{'value2':2}]
        p=numeric_positions(tok,item);text=tok.apply_chat_template([{'content':item['prompt']}])+compact(item['answer'])+'~'
        self.assertEqual(''.join(text[i+1] for i in p),'192')

    def test_numeric_loss_is_shifted_and_ignores_other_tokens(self):
        logits=torch.zeros(2,6,5);ids=torch.tensor([[0,1,2,3,4,0],[1,2,3,4,0,1]])
        logits[0,2,3]=4;logits[1,3,0]=5
        expected=torch.stack([logits[0,2].log_softmax(-1)[3],logits[1,3].log_softmax(-1)[0]])
        torch.testing.assert_close(numeric_logp(logits,ids,[[2],[3]]),expected)
        logits[:,0]=100;torch.testing.assert_close(numeric_logp(logits,ids,[[2],[3]]),expected)

    def test_protected_adapters_leave_module_weights_exact_after_update_and_merge(self):
        torch.manual_seed(7)
        cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        base=Qwen3ForCausalLM(cfg);original=copy.deepcopy(base.model.layers[0].mlp.state_dict())
        active_original=copy.deepcopy(base.model.layers[1].mlp.state_dict())
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
        names=protect_adapters(model,['0:mlp']);self.assertTrue(names)
        trainable=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(trainable,lr=.1)
        ids=torch.tensor([[3,4,5,6],[6,7,8,9]]);model(ids,labels=ids,use_cache=False).loss.backward();opt.step()
        merged=model.merge_and_unload()
        for k,v in original.items():torch.testing.assert_close(merged.model.layers[0].mlp.state_dict()[k],v,atol=0,rtol=0)
        self.assertTrue(any(not torch.equal(v,merged.model.layers[1].mlp.state_dict()[k]) for k,v in active_original.items()))
        with self.assertRaises(ValueError):module_prefix('0:residual')


if __name__=='__main__':unittest.main()
