import torch
import torch.nn as nn

"""
arr = nn.Embedding(3,4)

model = nn.Sequential()
model.add_module('emb_luts', nn.Embedding(3,4))

print(arr)
print(model[0][0])
"""

model = nn.Sequential(nn.Embedding(3,4,sparse=True))

temp = torch.ones(5,2,1,dtype=torch.long)
out = model(temp)

print(temp)
print(out,out.shape)