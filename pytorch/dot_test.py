import torch
import torch.nn as nn

a = torch.randn(1,5)
b = torch.randn(1,5)

print(a)
print(b)

print(torch.dot(a,b))
print(a.dot(b))