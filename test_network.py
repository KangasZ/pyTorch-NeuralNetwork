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

def test_network():
    x = torch.tensor([[1, 2], [3, 4]], device=DEVICE, dtype=DTYPE)
    w1 = torch.tensor([[-1, 1], [2, 1], [-2, 2]], device=DEVICE, dtype=DTYPE)
    b1 = torch.tensor([[1], [-5], [3]], device=DEVICE, dtype=DTYPE)
    w2 = torch.tensor([[1, -1, 1], [0, -1, 0]], device=DEVICE, dtype=DTYPE)
    b2 = torch.tensor([[-5], [2]], device=DEVICE, dtype=DTYPE)
    nntest = NeuralNetwork.Network(2, 2, DEVICE)
    actual = torch.tensor([[1, 3], [2, 4]], device=DEVICE, dtype=DTYPE)
    nntest.add_linear(w1, b1)
    nntest.add_linear(w2, b2)
    assert torch.all(nntest.forward(x, actual) == torch.tensor([[5,2],[2,-1]])), "Not expected output"
    assert nntest.loss == 21, "Non-expected loss"

def test_relu():
    x = torch.tensor([[-5, -2], [-3, -4], [1,1]], device=DEVICE, dtype=DTYPE)
    nntest = NeuralNetwork.Network(3, 3, DEVICE)
    actual = torch.tensor([[0, 0], [0, 0], [1,1]], device=DEVICE, dtype=DTYPE)
    nntest.add_relu()
    l = nntest.forward(x, actual)
    assert torch.all(l == actual), "Not expected output"
    assert nntest.loss == 0, "Non-expected loss"

def test_softmax1():
    x = torch.tensor([[10], [0], [0]], device=DEVICE, dtype=DTYPE)
    nntest = NeuralNetwork.Network(3, 3, DEVICE, loss="ce")
    actual = torch.tensor([[1], [0], [0]], device=DEVICE, dtype=DTYPE)
    nntest.add_softmax()
    l = nntest.forward(x, actual)
    assert torch.all(torch.round(l, decimals=1) == actual), "Not expected output"
    assert nntest.accuracy(x,actual) == 1, "Non-expected accuracy"

def test_softmax2():
    x = torch.tensor([[10, 0], [0, 5], [0,5]], device=DEVICE, dtype=DTYPE)
    nntest = NeuralNetwork.Network(3, 3, DEVICE, loss="ce")
    actual = torch.tensor([[1, 0], [0, 0.5], [0, .5]], device=DEVICE, dtype=DTYPE)
    nntest.add_softmax()
    l = nntest.forward(x, actual)
    assert torch.all(torch.round(l, decimals=1) == actual), "Not expected output"
    assert nntest.accuracy(x,actual) == 1, "Non-expected accuracy"

def test_network2_and_graddoesntscrash():
    x = torch.tensor([[1], [3], [2]], device=DEVICE, dtype=DTYPE)
    w1 = torch.tensor([[-1, 1, .5], [-1, 1, -1], [-2, 2, 1]], device=DEVICE, dtype=DTYPE)
    b1 = torch.tensor([[-2], [0], [-5.5]], device=DEVICE, dtype=DTYPE)
    w2 = torch.tensor([[1, -1, 1], [0, -1, 0]], device=DEVICE, dtype=DTYPE)
    b2 = torch.tensor([[-5], [2]], device=DEVICE, dtype=DTYPE)
    nntest = NeuralNetwork.Network(3, 2, DEVICE, regularization_factor=0.01)
    actual = torch.tensor([[-3], [1]], device=DEVICE, dtype=DTYPE)
    nntest.add_linear(w1, b1, regularization=True)
    nntest.add_linear(w2, b2)
    l = nntest.forward(x, actual)
    assert nntest.loss == 1.25
    assert torch.all(torch.tensor([[-3.5], [2]]) == l)
    nntest.backward()
    assert nntest.objective_function.output == 1.3925



def test_one_reg(verbose=False):
    reg_fact = 0.01
    x = torch.tensor([[1], [3], [2]], device=DEVICE, dtype=DTYPE)
    w1 = torch.tensor([[-1, 1, .5], [-1, 1, -1], [-2, 2, 1]], device=DEVICE, dtype=DTYPE)
    b1 = torch.tensor([[-2], [0], [-5.5]], device=DEVICE, dtype=DTYPE)
    actual = torch.tensor([[-4], [12], [1]], device=DEVICE, dtype=DTYPE)

    # NO EFFECT from regularization
    nntest = NeuralNetwork.Network(3, 3, DEVICE, regularization_factor=reg_fact)
    lin1 = nntest.add_linear(w1, b1, regularization=True)
    out = nntest.forward(x, actual)
    assert torch.all(out == torch.tensor([[1],[0],[0.5]]))
    assert nntest.loss == 169.25
    nntest.backward()
    assert nntest.objective_function.output == 169.3925
    assert torch.all(lin1.gradient == torch.tensor([[16],[-16],[28]]))
    assert torch.all(lin1.b.gradient == torch.tensor([[10],[-24],[-1]]))
    assert torch.all(lin1.W.gradient == torch.tensor([[9.9800,30.0200,20.0100],
                                                      [-24.0200, -71.9800, -48.0200],
                                                      [ -1.0400,  -2.9600,  -1.9800]],
                                                      dtype=torch.float64))


if __name__ == '__main__':
    test_network()
    test_relu()
    test_network2_and_graddoesntscrash()
    test_softmax1()
    test_softmax2()
    test_one_reg()