import torch
import torch.nn as nn
import torch.nn.functional as f

m = torch.randn(10,1,2)
m1 = f.dropout(m,p=0.5,training=True)

print(m)
print(m1)