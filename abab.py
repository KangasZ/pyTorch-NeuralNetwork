import torch
a = torch.tensor([[2,2],[4,4]],dtype=torch.float32)
b = torch.tensor([[1,2],[1,2]],dtype=torch.float32).sum(axis=0)
d = torch.div(a,b)
print(d)