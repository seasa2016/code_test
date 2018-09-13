import torch
import torch.nn as nn
import numpy as np

insts = [['a','b','c'],['a'],['a','b']]
max_len = max(len(inst) for inst in insts)


inst_data = np.array([
    inst + ['0'] * (max_len - len(inst))
    for inst in insts
])
print(inst_data)
inst_position = np.array([
    [pos_i+1 if(w_i != '0') else 0 for pos_i,w_i in enumerate(inst) ]
    for inst in inst_data
])
print(inst_position)