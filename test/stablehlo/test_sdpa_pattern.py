import math

import torch_xla.core.xla_model as xm
import torch
from torch_xla.experimental import tagging_utils
from torch_xla.stablehlo import exported_program_to_stablehlo
import torch.nn.functional as F


def sdpa_pattern(q, k, v, scale):
  return F.scaled_dot_product_attention(q, k, v, scale=scale)


class M(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x, y):
    q, k, v = x.split(128, dim=-2)
    attn_out = F.scaled_dot_product_attention(q, k, v, scale=0.25)
    q, k, v = y.split(128, dim=-2)
    attn_out2 = F.scaled_dot_product_attention(q, k, v, scale=4)
    return attn_out, attn_out2


q = torch.rand(32, 8, 128, 64)
k = torch.rand(32, 8, 128, 64)
v = torch.rand(32, 8, 128, 64)
attn_in = torch.concat((q, k, v), dim=-2)
attn_in2 = torch.concat((q, k, v), dim=-2)

sdpa_ep = torch.export.export(sdpa_pattern, (q, k, v, 9))
print(sdpa_ep)

m = M().eval()
args = (attn_in, attn_in2)
model_ep = torch.export.export(m, args)
pattern_args = (q, k, v)
model_ep = tagging_utils.mark_pattern(
    "sdpa_pattern",
    model_ep,
    sdpa_pattern,
    (q, k, v, 0.32),
    const_attr_trackers=[
        tagging_utils.ConstAttrTracker(
            "scale",
            pattern_arg_pos=3,
        ).track(0.1, 0.2)
    ],
)
model_ep.graph_module.graph.print_tabular()

shlo_bundle = exported_program_to_stablehlo(model_ep)
print(shlo_bundle.get_stablehlo_text())
