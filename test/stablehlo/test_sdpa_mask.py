import torch_xla.core.xla_model as xm
import torch
from torch_xla.experimental import tagging_utils
import torch.nn.functional as F
import numpy as np
import copy
import math

import os
import sys
sys.path.append("..")
os.environ['XLA_HLO_DEBUG'] = '1'


# Define the pattern in function with PyTorch building block(s)
def sdpa_pattern(q, k, v, mask, scale=9):
  return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


class SampleModel(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, q, k, v, y):
    q = q + 1
    k = k + 1
    v = v + 1
    mask = torch.gt(y, 0.5)
    out = F.scaled_dot_product_attention(q, k, v, mask, scale=0.25)
    return out


# Prepare the inputs for tracking pattern and model graph
q = np.random.randn(3, 1, 10, 10).astype(np.float32)
k = np.random.randn(3, 1, 10, 10).astype(np.float32)
v = np.random.randn(3, 1, 10, 10).astype(np.float32)
mask_float = np.random.randn(3, 1, 10, 10).astype(np.float32)
mask = mask_float > 0
q, k, v, mask, mask_float = [
    torch.tensor(t) for t in [q, k, v, mask, mask_float]
]

m = SampleModel().eval()
args = (q, k, v, mask_float)

# Export the model to FX graph
model_ep = torch.export.export(m, args)

pattern_args = (q, k, v, mask)
# Mark patterns in model graph
# tagging_utils.mark_pattern would do the following this
# 1. Export `sdpa_pattern` to FX graph with `pattern_args`
# 2. Use constant tracking algorithm to find out all non-tensor pattern inputs
# 3. Find out all occurances of `sdpa_pattern` FX in `model_ep` FX
# 4. Find out the non-tensor inputs to these occurances
# 5. Replace subgraphs (occurances) with pattern + tensor input annotators + const attrs
# model_ep = tagging_utils.mark_pattern(
#     "sdpa_pattern",
#     model_ep,
#     sdpa_pattern,
#     [pattern_args, pattern_args],
#     pattern_kwargs=[{"scale": 0.1}, {"scale":0.2}],
# )
# model_ep = tagging_utils.mark_pattern(
#     "sdpa_pattern",
#     model_ep,
#     sdpa_pattern,
#     pattern_args,
#     const_attr_trackers=[
#         tagging_utils.ConstAttrTracker("scale_sqrt", pattern_arg_pos=4)
#             .track(0.1, math.sqrt(0.1))
#             .track(0.2, math.sqrt(0.2))
#             .track(0.3, math.sqrt(0.3))
#     ]
# )
# Run the model_ep FX with torch_xla to record stablehlo
# The tensor input annotators are torch-xla specific funtion.
# In Python, they do nothing and behave like identities. But in torch_xla they will
# insert metadata to the XLA graph node serialized as JSON when dumping StableHLO
args = tuple(i.to(xm.xla_device()) for i in args if hasattr(i, "to"))
res = model_ep(*args)

stablehlo = xm.get_stablehlo([res])
print(stablehlo)
