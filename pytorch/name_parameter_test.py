import torch
import torch.nn as nn
"""
model = nn.Sequential(nn.Linear(1,2))

for i in model.named_parameters():
    print(i)
"""
class test(nn.Module):
    def __init__(self):
        super(test,self).__init__()

        self.linear2 = nn.Linear(1,5)
        self.linear1 = nn.Linear(5,1)
        
    def forward(self,x):
        pass
        x = self.linear1(x)
        x = self.linear2(x)

class t(nn.Module):
    def __init__(self):
        super(t,self).__init__()
        self.test123123 = test()

model = t()
for i in model.named_parameters():
    print(i)
print('-'*10)
print(model.parameters)
print('-'*10)
print(model)
print('-'*10)

