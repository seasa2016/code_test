from __future__ import print_function
import torch
import torch.nn as nn

xx = Variable(torch.randn(1,1), requires_grad = True)
yy = 3*xx
zz = yy**2

yy.register_hook(print)
zz.backward()