import torch
import torch.nn as nn
import torch.optim as optim
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

class fnn(nn.Module):
    def __init__(self):
        super(fnn,self).__init__()
        self.linear
model = nn.Sequential(nn.Linear(5,10),nn.Linear(10,5))
model = model.cuda()

optim = optim.Adam(model.parameters())

torch.save(optim.state_dict(),'qq')
optim.load_state_dict(torch.load('qq'))

for state in optim.state:
    print(state)