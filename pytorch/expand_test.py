import torch
import torch.nn as nn

a = torch.zeros([2,5])
b = torch.ones([2,1])
print(a)
print(b)
b = b.expand_as(a)
print(b)