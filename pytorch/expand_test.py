import torch
import torch.nn as nn

a = torch.range(1,5)
z = torch.zeros(2,5,5)

print(a)
print(z)
a = a.expand_as(z)
print(a)