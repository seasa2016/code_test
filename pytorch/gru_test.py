import torch
import torch.nn as nn

class test_gru(nn.Module):
    def __init__(self,n_layers=1):
        super(test_gru,self).__init__()

        self.gru = nn.GRU(3,3,n_layers)
    
    def forward(self,input,hidden):
        output,hidden = self.gru(input,hidden)

        return output,hidden

model = test_gru(n_layers=2).cuda()

input,hidden = torch.randn(1,1,3).cuda(),torch.randn(2,1,3).cuda()

output,hidden = model(input,None)
print(output,hidden)