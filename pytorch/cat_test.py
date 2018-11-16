import torch
import torch.nn as nn

a = torch.tensor([[1,2,3]])
b = torch.tensor([[1,2,3]])

print(a.shape)
print(torch.cat([a,b],dim=-1))
