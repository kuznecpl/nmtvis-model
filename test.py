# test_cuda.py
import torch
from datetime import datetime

for i in range(10):
    x = torch.randn(10, 10, 10, 10)
    t1 = datetime.now()
    x.cuda()
    print(i, datetime.now() - t1)
