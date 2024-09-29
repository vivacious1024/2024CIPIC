import numpy as np
import torch
import torch.nn

a=np.array([[1,2,3],[300,200,100]])
for i in range(a.shape[0]):
    b=a[i,:]
    a[i,:]=(a[i,:]-np.mean(a[i,:]))/np.std(a[i,:])

