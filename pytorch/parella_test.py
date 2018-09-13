import torch
import torch.nn as nn

class DataParallelModel(nn.Module):
    def __init__(self):
        super(DataParallelModel,self).__init__()
        self.block1 = nn.Linear(10,20)

        self.block2 = nn.Linear(20,20)
        self.block2 = nn.DataParallel(self.block2)

        self.block3 = nn.Linear(20,20)
    
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

model = DataParallelModel()
model = model.cuda()

for k in model.parameters():
    print(k)