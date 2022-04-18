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


def test_one_layer_grad(verbose = False):
    x = torch.tensor([[1], [3], [2]], device=DEVICE, dtype=DTYPE)
    w1 = torch.tensor([[-1, 1, .5], [-1, 1, -1], [-2, 2, 1]], device=DEVICE, dtype=DTYPE)
    b1 = torch.tensor([[-2], [0], [-5.5]], device=DEVICE, dtype=DTYPE)
    actual = torch.tensor([[-4], [12], [1]], device=DEVICE, dtype=DTYPE)

    nntest = NeuralNetwork.Network(3, 3, DEVICE)
    lin1 = nntest.add_linear(w1, b1)
    out = nntest.forward(x, actual)
    nntest.backward()

    x_b = torch.tensor([[1], [3], [2]], device=DEVICE, dtype=DTYPE)
    w1_b = torch.tensor([[-1, 1, .5], [-1, 1, -1], [-2, 2, 1]], device=DEVICE, dtype=DTYPE, requires_grad=True)
    b1_b = torch.tensor([[-2], [0], [-5.5]], device=DEVICE, dtype=DTYPE, requires_grad=True)
    actual_b = torch.tensor([[-4], [12], [1]], device=DEVICE, dtype=DTYPE)
    l2_b = (((torch.matmul(w1_b, x_b) + b1_b) - actual_b) ** 2)
    l2_b.retain_grad()
    loss_b = l2_b.sum()
    #print(loss_b)
    loss_b.backward()
    #print(l2_b.grad)
    if verbose:
        print("Grad from network:")
        print("Weight:", lin1.W.gradient)
        print("Bias:", lin1.b.gradient)
        print("Grad from backwrd():")
        print("Weight:", w1_b.grad)
        print("Bias:", b1_b.grad)

    assert torch.all(lin1.W.gradient == w1_b.grad), "Not expected gradient"
    assert torch.all(lin1.b.gradient == b1_b.grad), "Not expected gradient"
    assert loss_b ==nntest.loss, "Not expected gradient"


def test_backward_doesnt_work():
    x = torch.tensor([[1], [3], [2]], device=DEVICE, dtype=DTYPE)
    w1 = torch.tensor([[-1, 1, .5], [-1, 1, -1], [-2, 2, 1]], device=DEVICE, dtype=DTYPE)
    b1 = torch.tensor([[-2], [0], [-5.5]], device=DEVICE, dtype=DTYPE)
    actual = torch.tensor([[-4], [12], [1]], device=DEVICE, dtype=DTYPE)

    nntest = NeuralNetwork.Network(3, 3, DEVICE)
    lin1 = nntest.add_linear(w1, b1)
    assert nntest.backward() is False
    out = nntest.forward(x, actual)
    assert nntest.backward() is True
    assert nntest.backward() is False
    nntest.inference(x)
    assert nntest.backward() is False


def test_one_layer_grad_with_reg(verbose = False):
    reg_fact = 0.1
    x = torch.tensor([[1], [3], [2]], device=DEVICE, dtype=DTYPE)
    w1 = torch.tensor([[-1, 1, .5], [-1, 1, -1], [-2, 2, 1]], device=DEVICE, dtype=DTYPE)
    b1 = torch.tensor([[-2], [0], [-5.5]], device=DEVICE, dtype=DTYPE)
    actual = torch.tensor([[-4], [12], [1]], device=DEVICE, dtype=DTYPE)

    # NO EFFECT from regularization
    nntest = NeuralNetwork.Network(3, 3, DEVICE, regularization_factor=reg_fact)
    lin1 = nntest.add_linear(w1, b1, regularization=True)
    out = nntest.forward(x, actual)
    nntest.backward()

    x_b = torch.tensor([[1], [3], [2]], device=DEVICE, dtype=DTYPE)
    w1_b = torch.tensor([[-1, 1, .5], [-1, 1, -1], [-2, 2, 1]], device=DEVICE, dtype=DTYPE, requires_grad=True)
    b1_b = torch.tensor([[-2], [0], [-5.5]], device=DEVICE, dtype=DTYPE, requires_grad=True)
    actual_b = torch.tensor([[-4], [12], [1]], device=DEVICE, dtype=DTYPE)
    l2_b = (((torch.matmul(w1_b, x_b) + b1_b) - actual_b) ** 2)
    loss_b = l2_b.sum()
    reg = (reg_fact * ((w1_b ** 2).sum()))
    n = loss_b + reg
    n.backward()
    if verbose:
        print("Grad from network:")
        print("Weight:", lin1.W.gradient)
        print("Bias:", lin1.b.gradient)
        print("Grad from backwrd():")
        print("Weight:", w1_b.grad)
        print("Bias:", b1_b.grad)

    assert torch.all(lin1.W.gradient == w1_b.grad), "Not expected gradient"
    assert torch.all(lin1.b.gradient == b1_b.grad), "Not expected gradient"
    assert loss_b == nntest.loss, "Not expected gradient"


if __name__ == '__main__':

    test_backward_doesnt_work()
    test_one_layer_grad()
    test_one_layer_grad_with_reg()