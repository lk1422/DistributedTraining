import torch
import torch.nn as nn
from torchinfo import summary
from typing import Tuple, List

def wrap_module(module : nn.Module, dependencies : List[str], 
                outgoing_variables : List[str]):
    module.dependencies = []
    module.out_variables = []
    for dependency in dependencies:
        module.dependencies.append(dependency)
    for out_var in outgoing_variables:
        module.out_variables.append(out_var)
        setattr(module, out_var, None)
    return module

class wrapped_linear(nn.Module):
    def __init__(self, in_features=None, out_features=None, dependencies=[],
                 out_variables=[]):
        super(wrapped_linear,self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dependencies = dependencies
        self.forward_order = dependencies
        self.out_variables = out_variables
    
    def forward(self, x, noise=None):
        if(len(self.dependencies) != 0):
            if self.dependencies[0] == "random_noise":
                x = x + noise
                lin_out = self.linear(x) 
            elif self.dependencies[0] == "res_1":
                lin_out = self.linear(x) + noise
        if(len(self.out_variables) == 1):
            return lin_out, [(self.out_variables[0], lin_out)]
        return self.linear(x)


