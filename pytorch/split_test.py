import torch


a = torch.rand(4,6)
print(a)

for q in torch.split(a,2):
    print(q)