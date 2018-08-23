import torch
import torch.nn as nn
import numpy as np

size = 5
attn_shape = (1,size,size)

subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
subsequent_mask = torch.from_numpy(subsequent_mask)

print(subsequent_mask)
