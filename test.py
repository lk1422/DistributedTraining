import torch
import torch.nn as nn
from bossman import BossMan
import Server
import asyncio
import matplotlib.pyplot as plt
from typing import Tuple, List


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
        else:
            lin_out = self.linear(x)
        if(len(self.out_variables) == 1):
            return lin_out, [(self.out_variables[0], lin_out)]
        return self.linear(x)


TEST_RESIDUAL_MODEL = [
        (nn.Linear, {'in_features':100, 'out_features':512}, (32,  100)),
        (wrapped_linear, {
            'in_features':512, 
            'out_features':1024,
            'dependencies':["random_noise"],
            'out_variables': ["res_1"]
        }, [(32, 512), (32,512)]),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32,1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32,1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32,1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32,1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32,1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32,1024)),
        (wrapped_linear, {
            'in_features':1024, 
            'out_features':1024,
            'dependencies':["res_1"],
            'out_variables': []
        }, [(32, 1024), (32,1024)]),
        (nn.Linear, {'in_features':1024, 'out_features':1}, (32, 1024))
    ]

TEST_RESIDUAL_SMALL = [
        (nn.Linear, {'in_features':1, 'out_features':512}, (32,  1)),
        (nn.ReLU,   {}, (32,512)),
        (wrapped_linear, {
            'in_features':512, 
            'out_features':512,
            'out_variables': ["res_1"]
            }, [(32, 512)]),
        (nn.ReLU,   {}, (32,512)),
        (nn.Linear, {'in_features':512, 'out_features':512}, (32,512)),
        (nn.ReLU,   {}, (32,512)),
        (wrapped_linear, {
            'in_features':512, 
            'out_features':512,
            'dependencies':["res_1"],
            'out_variables': []
        }, [(32, 512), (32,512)]),
        (nn.ReLU,   {}, (32,512)),
        (nn.Linear, {'in_features':512, 'out_features':1}, (32, 512)),
        (nn.Sigmoid,   {}, (32,1))
    ]


TEST_SIMPLE_MODEL = [
        (nn.Linear, {'in_features':1, 'out_features':512}, (32,  1)),
        (nn.ReLU, {}, (32,512)),
        (nn.Linear, {'in_features':512, 'out_features':1024}, (32,512)),
        (nn.ReLU, {}, (32,1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1}, (32, 1024)),
        (nn.Sigmoid,   {}, (32,1))
    ]


