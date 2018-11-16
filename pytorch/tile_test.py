import torch

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1)
    print('1',x) 
    x = x.transpose(0, 1)
    print('2',x)
    x = x.repeat(count, 1)
    print('3',x)
    x = x.transpose(0, 1)
    print('4',x)
    x = x.contiguous()
    print('5',x)
    x = x.view(*out_size)
    print('6',x)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

a = torch.tensor([[1,2,3],[4,5,6]])

temp = tile(a,2,1)
print(temp)