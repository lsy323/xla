import os
import timm
import torch
import torch.nn.functional as F
from torch.utils import _pytree as pytree
import torch_xla.core.xla_model as xm
from torch_xla.experimental import tagging_utils
from torch_xla.stablehlo import XLAExportInterpreter
from torch_xla.stablehlo import exported_program_to_stablehlo
import math

os.environ["TIMM_FUSED_ATTN"] = "0"
os.environ["XLA_HLO_DEBUG"] = "1"


def fused_sdpa_pattern(q, k, v, attn_drop, scale):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop, scale=scale)


def non_fused_sdpa_pattern(q, k, v, attn_drop, scale):
    q = q * scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = torch.nn.Dropout(attn_drop)(attn)
    x = attn @ v
    return x


model = timm.create_model("vit_small_patch16_224")
args = (torch.randn(1, 3, 224, 224),)
m_res = model(*args)
exported_model = torch.export.export(model, args)
q = torch.rand(32, 8, 128, 64)
k = torch.rand(32, 8, 128, 64)
v = torch.rand(32, 8, 128, 64)

exported_model.graph_module.graph.print_tabular()

exported_model = tagging_utils.mark_pattern(
    "sdpa_pattern",
    exported_model,
    fused_sdpa_pattern,
    (q, k, v, 0, 0.32),
    const_attr_trackers=[
        tagging_utils.ConstAttrTracker(
            "scale",
            transform=math.sqrt,
            inverse_transform=lambda x: x**2,
            pattern_arg_pos=4,
        ).track(0.1, 0.2)
    ],
)
exported_model = tagging_utils.mark_pattern(
    "sdpa_pattern",
    exported_model,
    non_fused_sdpa_pattern,
    (q, k, v, 0, 0.32),
    const_attr_trackers=[
        tagging_utils.ConstAttrTracker(
            "scale",
            pattern_arg_pos=4,
        ).track(0.1, 0.2)
    ],
)

shlo_module = exported_program_to_stablehlo(exported_model)
# print(shlo_module.get_stablehlo_text())
# print(shlo_module.get_stablehlo_bytecode())
res = shlo_module(*args)
print(res.shape, m_res.shape)
print(torch.mean((res[0] - m_res[0]) ** 2))  # not small
