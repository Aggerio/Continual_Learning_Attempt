"""Native gated-MLP channel banks, used only after held-out localization passes."""
import argparse
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from continual import nonlinear_layers as training
from continual.worker import load_runtime


class GatedBank(nn.Module):
    def __init__(self,hidden,channels):
        super().__init__()
        self.gate=nn.Linear(hidden,channels,bias=False)
        self.up=nn.Linear(hidden,channels,bias=False)
        self.down=nn.Linear(channels,hidden,bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self,x):
        return self.down(F.silu(self.gate(x.float()))*self.up(x.float())).to(x.dtype)


def attach(model,layers,channels):
    if hasattr(model,'temporary_layers'):raise ValueError('Already expanded')
    if len(set(layers))!=len(layers) or any(i<0 or i>=len(model.model.layers) for i in layers):raise ValueError('Invalid sites')
    model.requires_grad_(False)
    banks=nn.ModuleDict({str(i):GatedBank(model.config.hidden_size,channels) for i in layers})
    banks.to(device=model.device,dtype=torch.float32);model.add_module('temporary_layers',banks)
    model.temporary_bypass=set();handles=[]
    for i in layers:
        def hook(_module,inputs,output,index=i):
            return output if index in model.temporary_bypass else output+banks[str(index)](inputs[0])
        handles.append(model.model.layers[i].mlp.register_forward_hook(hook))
    model.temporary_handles=handles
    return banks


def load_expanded(source,config,teacher=None):
    from safetensors.torch import load_file
    _,tokenizer,model=load_runtime(source)
    attach(model,config['insertion_layers'],config['channels_per_site'])
    if teacher:model.temporary_layers.load_state_dict(load_file(str(Path(teacher)/'layers.safetensors')))
    return tokenizer,model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['train','cache','eval'])
    for name in ['config','source','data','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=['teacher','output']);p.add_argument('--role',choices=['old','new'])
    p.add_argument('--teacher');p.add_argument('--schedule');p.add_argument('--old-cache');p.add_argument('--new-cache')
    p.add_argument('--bypass',default='');args=p.parse_args()
    # Reuse the audited optimizer, token schedule, target cache and evaluator,
    # changing only the expanded architecture in this fresh process.
    training.load_expanded=load_expanded
    {'train':training.train,'cache':training.cache_targets,'eval':training.evaluate}[args.action](args,training.read(args.config))


if __name__=='__main__':main()
