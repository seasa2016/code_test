import torch
import torch.nn as nn

class test(nn.Module):
    def __init__(self):
        super(test,self).__init__()

        self.l1 = nn.Linear(10,20)
    
    def forward(self,x):
        output = self.l1(x)

        return output

gen = nn.Sequential(nn.Linear(20,10))

model = test()
model.gen = gen

model.to("cuda")

arr = torch.randn(2,10).cuda()
output = model(arr)

print(output)