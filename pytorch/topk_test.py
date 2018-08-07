import torch
import torch.nn as nn

a = torch.tensor([1,2,6,3,4,5],dtype=torch.float,requires_grad=False)
b = torch.randn(6,dtype=torch.float,requires_grad=True)
print(a.shape,b.shape)

learning_rate = 0.005
criterion = nn.MSELoss()
ans = torch.tensor([42],dtype=torch.float)
c = torch.dot(a,b)
print(c)

for epoch in range(50):

    c = torch.dot(a,b)
    loss = criterion(c,ans)
    
    print(loss)

    loss.backward()
    
    with torch.no_grad():
        b -= learning_rate * b.grad
        b.grad.data.zero_()
    
