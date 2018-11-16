import torch

padding_idx = 0
tgt_dict_size = 5
label_smoothing = 0.1


smoothing_value = label_smoothing / (tgt_dict_size - 2)
confidence = 1 - label_smoothing

one_hot = torch.full((tgt_dict_size,),smoothing_value)
one_hot[padding_idx] = 0

def test(target,mask):
    model_prob = one_hot.repeat(target.shape[0],1)
    model_prob.masked_fill_((target == padding_idx).unsqueeze(1),0)
    
    for i in range(mask.shape[0]):
        model_prob[i][mask[i] ] = 0

    model_prob.scatter_(1,target.unsqueeze(1),confidence)

    print(model_prob)

while(1):
    arr = [int(_) for _ in input().strip().split()]
    mask = [int(_) for _ in input().strip().split()]
    
    test(torch.tensor(arr),torch.tensor(mask))