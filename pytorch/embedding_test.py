from __future__ import print_function
import torch 
import torch.nn as nn
import math
"""
#backprop
# an Embedding module containing 10 tensors of size 3
aa = nn.Embedding(5, 3,padding_idx=0)
# a batch of 2 samples of 4 indices each
i = torch.LongTensor([0,1,2,3,4])
#print(aa.weight)
a = aa(i)
print(a.data)
print(aa.weight)

s = torch.sum(a)
loss = s-1
loss.backward()
print(aa.weight._grad)
"""

"""
#assignment
aa = nn.Embedding(5,3)
t = torch.FloatTensor([1,2,3])
print(aa.weight)
aa.weight[0] = t
print(aa.weight)
"""

shape = [4,4]
pe = torch.zeros(shape)
position = torch.arange(0, shape[0]).unsqueeze(1)
div_term = torch.exp((torch.arange(0, shape[1], 2,dtype=torch.float) *-(math.log(10000.0,math.e) / shape[1])).float())
print(div_term)
pe[:, 0::2] = torch.sin(position.float() * div_term)
pe[:, 1::2] = torch.cos(position.float() * div_term)
pe = pe.unsqueeze(1)
