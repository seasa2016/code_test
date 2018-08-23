import torch
import torch.nn as nn

arr = torch.randn(2,3)
print(arr)

qq=arr.mean(-1,keepdim=True)
qq1=arr.mean(-1,keepdim=False)
print(qq)
print(qq1)