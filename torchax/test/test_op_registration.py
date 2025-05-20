import torch
import torchax

env = torchax.default_env()

t = torch.rand(3)
with env:
    # breakpoint()
    t = t.to('jax')
    t2 = torch.zeros(3, device='cpu')
    breakpoint()
    print(t)
    print(t2)
