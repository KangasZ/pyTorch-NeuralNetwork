import torch
import numpy as np
import warnings
import os.path

import NeuralNetwork
import Client

#With this block, we don't need to set device=DEVICE for every tensor.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.cuda.set_device(0)
     torch.set_default_tensor_type(torch.cuda.FloatTensor)
     print("Running on the GPU")
else:
     print("Running on the CPU")

def create_linear_training_data(training_points):
    """
    This method simply rotates points in a 2D space.
    Be sure to use L2 regression in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a numpy array where columns are training samples and
             y is a numpy array where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, training_points))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    y = torch.cat((-x2, x1), axis=0)
    return x, y

if __name__ == '__main__':
    x, y = create_linear_training_data(1000)
    epochs = 30

    nn = NeuralNetwork.Network(x.shape[0], y.shape[0], dtype=torch.float32, loss="l2", regularization_factor=0.01, learning_rate=0.001)
    lin3 = nn.add_linear_generated(num_nodes=2, w=0.1, wo=0, b=0, bo=0, regularization=True)


    client = Client.Client(nn, x, y)
    loss, acc = client.validation_stats()
    tloss, tacc = client.train_stats()
    vloss, vacc = client.validation_stats()
    print("E: -1", "\ttL:", tloss, "\ttA:", tacc, "\tvL:", vloss, "\tvA:", vacc)
    train_data = client.train(epochs, 1, verbose=True)



