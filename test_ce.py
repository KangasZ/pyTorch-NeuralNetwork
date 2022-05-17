import torch
import NeuralNetwork

#With this block, we don't need to set device=DEVICE for every tensor.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.cuda.set_device(0)
     torch.set_default_tensor_type(torch.cuda.FloatTensor)
     print("Running on the GPU")
else:
     print("Running on the CPU")
DTYPE = torch.float32

def test_ce_1(verbose=False):
    x = torch.tensor([[0.998], [0.001], [0.001]], dtype=DTYPE)
    nntest = NeuralNetwork.Network(3, 3, loss="ce")
    sm = nntest.add_softmax()
    actual = torch.tensor([[1], [0],[0]], dtype=DTYPE)

    l = nntest.forward(x, actual)
   # print(sm.output)
    #print(nntest._loss.output)
    nntest.backward()
    #print("Softmax...", sm.output)


    x_b = torch.tensor([[0.998], [0.001], [0.001]], dtype=DTYPE, requires_grad=True)
    actual_b = torch.tensor([[1], [0],[0]], dtype=DTYPE)
    softmax = torch.div(torch.exp(x_b), torch.exp(x_b).sum(axis=0))
    #print(softmax)
    ce = (actual_b * torch.log(softmax + 1E-7))
    ce = (ce.sum(axis=0) * -1).mean()
    softmax.retain_grad()
    x_b.retain_grad()
    ce.backward()
    if verbose:
        print("softmax grad", sm.gradient)
        print("loss grad", nntest._loss.gradient)
        print("Backward cald by torch")
        print("softmax grad", x_b.grad)
        print("loss grad", softmax.grad)


def test_ce_2(verbose=False):
    x = torch.tensor([[1], [0.6], [0.3]], dtype=DTYPE)
    nntest = NeuralNetwork.Network(3, 3, loss="ce")
    sm = nntest.add_softmax()
    actual = torch.tensor([[1], [0],[0]], dtype=DTYPE)

    l = nntest.forward(x, actual)
   # print(sm.output)
    #print(nntest._loss.output)
    nntest.backward()
    #print("Softmax...", sm.output)


    x_b = torch.tensor([[1], [0.6], [0.3]], dtype=DTYPE, requires_grad=True)
    actual_b = torch.tensor([[1], [0],[0]], dtype=DTYPE)
    softmax = torch.div(torch.exp(x_b), torch.exp(x_b).sum(axis=0))
    #print(softmax)
    ce = (actual_b * torch.log(softmax + 1E-7))
    ce = (ce.sum(axis=0) * -1).mean()
    softmax.retain_grad()
    x_b.retain_grad()
    ce.backward()
    if verbose:
        print("softmax grad", sm.gradient)
        print("loss grad", nntest._loss.gradient)
        print("Backward cald by torch")
        print("softmax grad", x_b.grad)
        print("loss grad", softmax.grad)


def test_ce_3():
    """
    THIS WILL NOT WORK, STOP IT
    :return:
    """
    x = torch.tensor([[1,5], [0.6,2], [0.3,1]], dtype=DTYPE)
    nntest = NeuralNetwork.Network(3, 3, loss="ce")
    sm = nntest.add_softmax()
    actual = torch.tensor([[1], [0],[0]], dtype=DTYPE)

    l = nntest.forward(x, actual)
   # print(sm.output)
    #print(nntest._loss.output)
    nntest.backward()
    #print("Softmax...", sm.output)

    print("softmax grad", sm.gradient)
    print("loss grad", nntest._loss.gradient)


    x_b = torch.tensor([[1,5], [0.6,2], [0.3,1]], dtype=DTYPE, requires_grad=True)
    actual_b = torch.tensor([[1], [0],[0]], dtype=DTYPE)
    softmax = torch.div(torch.exp(x_b), torch.exp(x_b).sum(axis=0))
    #print(softmax)
    ce = (actual_b * torch.log(softmax + 1E-7))
    ce = (ce.sum(axis=0) * -1).mean()
    softmax.retain_grad()
    x_b.retain_grad()
    ce.backward()
    print("Backward cald by torch")
    print("softmax grad", x_b.grad)
    print("loss grad", softmax.grad)



if __name__ == '__main__':
    # Please VISUALLY check gradients!
    test_ce_1(verbose=True)
    test_ce_2(verbose=True)
    # test_ce_3() THIS TEST DOES NOT WORK - SOFTMAX WILL NOT WORK IN 3D