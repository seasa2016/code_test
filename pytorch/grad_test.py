import torch
import torch.nn as nn 

a = torch.tensor([1,2,3,4],requires_grad=True)
s = torch.sum(a)
loss = s - 5
loss.backward()


print(a.grad)