import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(3,1))

model.train()

ans = torch.ones(10,1,dtype=torch.float)
a = torch.randn(10,3)
output = model(a)

loss = torch.mean((output-ans) ** 2)
loss.backward()

for k in model.parameters():
    print(k.grad)

a = torch.randn(10,3)
output = model(a)

loss = torch.mean((output-ans) ** 2)
loss.backward()

for k in model.parameters():
    print(k.grad)