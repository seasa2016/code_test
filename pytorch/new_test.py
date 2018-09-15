import torch

arr = torch.tensor([1,2,3])
#new_zeros

buffer1 = arr[0].new(5).zero_()
buffer2 = arr[0].new_zeros(6)

print(buffer1)
print(buffer2)

