import torch
import torch.nn as nn

a = torch.rand((2,3,10),dtype=torch.float)
b = torch.rand((2,2,10),dtype=torch.float)
### matmal()
res1 = torch.matmul(a,b.transpose(1,2))
print(res1)
"""
...
[torch.FloatTensor of size 2x3x2]
"""
### bmm()
res2 = torch.bmm(a,b.transpose(1,2))
print(res2)
"""
...
[torch.FloatTensor of size 2x3x2]
"""
print(torch.eq(res1,res2))