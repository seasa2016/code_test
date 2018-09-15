import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

qq = 1


# Assuming optimizer has two groups.
def lambda1(epoch):
    print(qq)
    return 1

model = nn.Sequential(nn.Linear(5,1,bias=False))
if(torch.cuda.is_available()):
    model = model.cuda()
    print("use gpu")

optimizer = optim.SGD(model.parameters(),lr = 0.000005)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


for epoch in range(10000):

    x = torch.randn(100,5)
    if(torch.cuda.is_available()):
        x = x.cuda()
    ans = x.mean(dim=-1,keepdim=True)
    
    model.train()
    
    out = model(x)
    loss = F.mse_loss(out,ans)
    
    loss.backward()
    print(loss)
    scheduler.step()
    optimizer.step()
    
for i in model.named_parameters():
    print(i)