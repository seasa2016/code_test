import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.tensor([0,.5],dtype=torch.float)
a.requires_grad = True
b = torch.tensor([1,0],dtype=torch.float)

a_ = torch.tensor([[0.5,0.5]],dtype=torch.float)
a_.requires_grad = True
b_ = torch.tensor([0],dtype=torch.long)

print(a_.shape,b_.shape)

print(F.kl_div(a,b))
F.kl_div(a,b).backward()
print(a.grad)

F.nll_loss(a_,b_).backward()
print(a_.grad)
