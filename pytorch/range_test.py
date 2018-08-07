import torch
import torch.nn as nn

a = torch.range(0,5,requires_grad=True)
loss = torch.sum(a) - 5
loss.backward()
print(a.grad)