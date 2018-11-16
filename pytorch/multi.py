import torch
import torch.nn as nn

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        
    def forward(self):
        print('a')
        

class B(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self):
        print('b')
        
class C(B,A ):
    ''' Perform the reshape routine before and after a linear projection '''
    pass

qq = C()
qq()