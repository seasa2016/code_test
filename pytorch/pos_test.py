import torch
import torch.nn as nn
import math

max_len = 2
dim = 4

pe = torch.zeros(max_len, dim)
position = torch.arange(0, max_len).unsqueeze(1)
div_term = torch.exp((torch.arange(0, dim, 2,dtype=torch.float) * (math.log(10000.0,math.e) / dim)).float())
div_term1 = torch.pow(10000.0, torch.arange(0, dim, 2).float() / dim)

print(div_term)
print(div_term1)
"""
print(position)
print(div_term)
pe[:, 0::2] = torch.sin(position.float() * div_term)
pe[:, 1::2] = torch.cos(position.float() * div_term)
pe = pe.unsqueeze(1)

print(pe)
"""