from NeuralNetwork import Network
import math


class Client:
    def __init__(self, network: Network, data_x, data_y):
        self.network = network
        self.data_x = data_x
        self.data_y = data_y

    def train_epoch(self, minibatch_size):
        for i in range(0, (int(math.ceil(self.data_y.shape[1] / minibatch_size)))):
            t_x = self.data_x[:, i * minibatch_size:(i * minibatch_size) + minibatch_size]
            t_y = self.data_y[:, i * minibatch_size:(i * minibatch_size) + minibatch_size]
            self.network.forward(t_x, t_y)
            self.network.backward()
        acc = self.network.accuracy(self.data_x, self.data_y)
        loss = self.network.loss
        return loss, acc

    def inference(self, input):
        return self.network.inference(input)

    def train(self, num_epochs, minibatch_size):
        results = []
        for epoch in range(0, num_epochs):
            loss, acc = self.train_epoch(minibatch_size)
            results.append((epoch, loss, acc))
        return results

