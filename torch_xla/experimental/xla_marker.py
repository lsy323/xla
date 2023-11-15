import torch
import torch_xla
from torch.library import Library, impl

xla_pattern_marking_lib = Library("xla_pattern_marking", "DEF")

xla_pattern_marking_lib.define("tag_tensor(Tensor x, str tag) -> Tensor")


@impl(xla_pattern_marking_lib, "tag_tensor", "CompositeExplicitAutograd")
def tag_tensor(x, tag):
  return x


@impl(xla_pattern_marking_lib, "tag_tensor", "XLA")
def tag_tensor(x, tag):
  return torch_xla._XLAC._xla_add_tag(x, tag)
  # return x


@impl(xla_pattern_marking_lib, "tag_tensor", "Meta")
def tag_tensor(x, tag):
  return torch.empty_like(x)
