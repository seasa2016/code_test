import numpy as np
import torch 

a = torch.tensor([[2,1,3,4,9],[2,9,1,3,4]],dtype=torch.float)
print(a.max(1))