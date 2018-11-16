import torch
import torch.nn as nn

a = torch.zeros(2,2,3,dtype=torch.float)
a += torch.tensor([[1,1,1],[2,2,2]],dtype=torch.float).unsqueeze(1)

print(a.shape)
