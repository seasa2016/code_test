import torch

a = torch.rand(4)
b = torch.tensor([1,0,1,1],dtype=torch.float)
a.requires_grad = True
print(a.shape)
print(b.shape)

loss = torch.sum(a*b-1)


loss.backward()
print(a.grad)