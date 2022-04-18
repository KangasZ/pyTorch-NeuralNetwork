import torch
import NeuralNetwork

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu().
       https://d2l.ai/chapter_deep-learning-computation/use-gpu.html
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
DEVICE=try_gpu()
DTYPE = torch.float64

def test_gradients_doesnt_crash():
    x = torch.tensor([[1], [3], [2]], device=DEVICE, dtype=DTYPE)
    w1 = torch.tensor([[-1, 1, .5], [2, 1, -1], [-2, 2, 1]], device=DEVICE, dtype=DTYPE)
    b1 = torch.tensor([[1], [-5], [3]], device=DEVICE, dtype=DTYPE)
    w2 = torch.tensor([[1, -1, 1], [0, -1, 0]], device=DEVICE, dtype=DTYPE)
    b2 = torch.tensor([[-5], [2]], device=DEVICE, dtype=DTYPE)
    nntest = NeuralNetwork.Network(3, 2, DEVICE)
    actual = torch.tensor([[9], [3]], device=DEVICE, dtype=DTYPE)
    nntest.add_linear(w1, b1)
    nntest.add_linear(w2, b2)
    l = nntest.forward(x, actual)
    nntest.backward()
    nntest.accumulate_grad()