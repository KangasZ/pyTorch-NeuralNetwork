from NeuralNetwork import Network
import math

class Client:
    def __init__(self, network: Network, data_x, data_y, train_test_split=0.2):
        self.network = network
        self.data_x = data_x
        self.data_y = data_y
        #Todo, shuffle
        y = int(math.ceil(data_y.shape[1]*(1-train_test_split)))
        self.train_x = data_x[:,0:y]
        self.train_y = data_y[:,0:y]
        self.test_x = data_x[:,y:self.data_x.shape[1]]
        self.test_y = data_y[:,y:self.data_x.shape[1]]


    def train_epoch(self, minibatch_size):
        for i in range(0, (int(math.ceil(self.train_x.shape[1] / minibatch_size)))):
            for j in range(i * minibatch_size, (i * minibatch_size) + minibatch_size):
                if j < self.train_x.shape[1]:
                    t_x = self.train_x[:, j:j+1]
                    t_y = self.train_y[:, j:j+1]
                    self.network.forward(t_x, t_y)
                    self.network.backward()
                self.network.step()
                self.network.zero_grad()
                #print(self.network.loss)
        acc = self.network.accuracy(self.train_x, self.train_y)
        loss = self.network.loss
        return loss, acc

    def validation_stats(self):
        acc = self.network.accuracy(self.test_x, self.test_y)
        loss = self.network.loss
        return loss, acc

    def train_stats(self):
        acc = self.network.accuracy(self.train_x, self.train_y)
        loss = self.network.loss
        return loss, acc

    def inference(self, input):
        return self.network.inference(input)

    def train(self, num_epochs, minibatch_size, verbose=False):
        results = []
        for epoch in range(0, num_epochs):
            tloss, tacc = self.train_epoch(minibatch_size)
            vloss, vacc = self.validation_stats()
            if verbose:
                print("E:", epoch, "\ttL:", tloss, "\ttA:", tacc, "\tvL:", vloss, "\tvA:", vacc)
            results.append((epoch, tloss, tacc, vloss, vacc))
        return results

