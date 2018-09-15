import torch

a = torch.tensor([[1,2,3],[4,5,6]])

print(a,a.numel())

for i in a:
    print(i,i.numel())