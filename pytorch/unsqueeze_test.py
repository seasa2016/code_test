import torch
import torch.nn as nn

a = torch.tensor([1,2,3,4,5,6,7])
print(a.shape)
a = a.unsqueeze(0)
print(a.shape)