import torch 
import torch.nn as nn

class test(nn.Module):
    def __init__(self):
        super(test,self).__init__()
        self.a = nn.Linear(5,2)
    
    def forward(self,x):
        aa = self.a(x)
        print(aa.shape)
        aa = aa.view(-1)
        return aa

t = torch.ones([2,5])

model = test()
print(model(t))