import torch


arr = torch.rand(10,10)
print(arr)
idx = torch.tensor([(0,[1,2]),(1,[2,3])])
print(arr[idx])