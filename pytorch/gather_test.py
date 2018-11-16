import torch

a = torch.rand(3,4,5)
b = torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=torch.long)
print(a)
print()
for out,tgt in zip(a,b):
    tgt = tgt.unsqueeze(1)
    print(out.shape,tgt.shape)
    scores = out.data.gather(1, tgt)
    print(scores,tgt)
    scores.masked_fill_(tgt.eq(0), 0)
#
#b = b.unsqueeze(1)
#print(a)
#print(a.gather(1,b))
