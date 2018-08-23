import torch
import torch.nn

a = torch.randn(2,3)
b = torch.ones(3,1)

print(a)
print('-'*10)
print(b)
print('-'*10)
c = torch.matmul(a,b)
print(c)