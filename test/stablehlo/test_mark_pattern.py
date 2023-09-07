import torch_xla
import torch_xla.core.xla_model as xm
import torch
import unittest
import numpy as np
import torch_xla.experimental
from torch_xla.experimental import tagging_utils

def dual_output(x, y):
    return (x + y, x - y)

# Pattern to match in the exported graph.
def dual_output_pattern(x, y):
    return dual_output(x, y)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        exp_x = torch.exp(x)
        exp_y = torch.exp(y)
        out1, out2 = dual_output(exp_x, exp_y)
        out = torch.log(out1) + torch.log(out2)
        return out
    
m = M().eval()
args = (torch.rand(2, 3), torch.rand(2, 3))
pattern_args = (torch.rand(2, 3), torch.rand(2, 3))
model_ep = torch._export.export(m, args)

print("check exported module")
model_ep.graph_module.graph.print_tabular()
model_ep = tagging_utils.mark_pattern(model_ep, dual_output_pattern, pattern_args)
print("check exported module after tagged")
model_ep.graph_module.graph.print_tabular()

args = tuple(i.to(xm.xla_device()) for i in args if hasattr(i, "to"))
res = model_ep(*args)

stablehlo = xm.get_stablehlo([res])
print(stablehlo)

#########
def my_softmax(x, y):
    out = x + y
    out = torch.abs(out)
    out = torch.nn.LogSoftmax(dim=1)(out)
    out = torch.nn.ReLU6()(out)
    out = torch.log(out)
    return out


def my_inverse(x):
    return torch.div(1, x) + x

# Pattern to match in the exported graph.
def pattern(x, y):
    return my_softmax(x, y)


# Pattern to match in the exported graph.
def inverse_pattern(x):
    return my_inverse(x)

class M2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        exp_x = torch.exp(x)
        exp_y = torch.exp(y)
        exp_z = torch.exp(z)
        out1 = my_softmax(exp_x, exp_y)
        out2 = my_softmax(exp_y, exp_z)
        out1 = torch.abs(out1)
        out2 = torch.abs(out2)
        out1 = my_inverse(out1)
        out2 = my_inverse(out2)
        out = out1 + out2
        return out

m = M2().eval()
args = (torch.rand(2, 3), torch.rand(2, 3), torch.rand(2, 3))
model_ep = torch._export.export(m, args)
pattern_args = (torch.rand(2, 3), torch.rand(2, 3))
inverse_pattern_args = (torch.rand(2, 3),)
# Wrap the exported graph with tagging callbacks
model_ep = tagging_utils.mark_pattern(model_ep, pattern, pattern_args)

model_ep = tagging_utils.mark_pattern(model_ep, inverse_pattern, inverse_pattern_args)

args = tuple(i.to(xm.xla_device()) for i in args if hasattr(i, "to"))
res = model_ep(*args)

stablehlo = xm.get_stablehlo([res])

print(stablehlo)