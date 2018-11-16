import torch

a = torch.randn(6,requires_grad=True)

b = a.detach()
c = a.clone()
d = a.data
print(a,a.requires_grad)
print(b,b.requires_grad)
print(c,c.requires_grad)
print(d,d.requires_grad)
a[0] = -1
print('*'*10)
print(a,a.requires_grad)
print(b,b.requires_grad)
print(c,c.requires_grad)
print(d,d.requires_grad)