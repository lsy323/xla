import torch_xla.core.xla_model as xm
import torch
from torch_xla.experimental import tagging_utils
import torch.nn.functional as F


def sdpa_pattern(q, k, v, scale=9):
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
    [(q,k,v), (q,k,v)],
    pattern_kwargs=[{"scale": 9.0}, {"scale":0.25}],
)

args = tuple(i.to(xm.xla_device()) for i in args if hasattr(i, "to"))
res = model_ep(*args)

stablehlo = xm.get_stablehlo([res[0],res[1]])
print(stablehlo)
