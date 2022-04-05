from Layers import *

class Network:
    def __init__(self, input_rows, output_rows, device, learning_rate=0.01, regularization_factor=0, loss='l2'):
        self._forward = False
        self._backward = False

        self._regularizations = []
        self._layers = []

        self._irows = input_rows
        self._orows = output_rows
        self._device = device

        self._learning_rate = learning_rate
        self._regularization_factor = regularization_factor

        self._head = Input(self._irows)
        self._objective_function = None
        if loss == 'l2':
            self._loss = L2()
        elif loss == 'cross entropy' or loss == 'ce' or loss == 'crossentropy':
            self._loss = CrossEntropy()
        else:
            raise ValueError("Get a actual loss you monke")
        self.loss = 0
        self._tail = self._head

    def _add_layer(self, new_layer):
        """
        Internal method for adding layer to the tail parameters
        :param new_layer:
        :return:
        """
        new_layer.tail = self._tail
        self._tail.head = new_layer
        self._tail = new_layer

    def add_linear_generated(self, num_nodes, w=0.1, wo=0, b=0, bo=0, regularization=False):
        """
        Create a generated linear layer using preset addendums to rand
        :param num_nodes: Number of nodes in layer for this generation
        :param w: Weight weight
        :param wo: Weight bias
        :param b: Bias weight
        :param bo: Bias bias
        :param regularization: Boolean to use regularization
        :return:
        """
        columns = self._tail.out_s
        rows = num_nodes
        weights = (torch.rand((rows, columns), device=self._device, dtype=torch.float64) + wo)*w
        bias = (torch.rand((rows, 1), device=self._device, dtype=torch.float64) + bo)*b
        self.add_linear(weights, bias, True)

    def add_linear(self, weights_tensor, bias_tensor, regularization=False):
        """
        Creates a linear layer based on two given tensors. Will encapsulate these in param
        :param weights_tensor:
        :param bias_tensor:
        :param regularization:
        :return:
        """
        # Todo: check shape
        weights = Param(weights_tensor)
        bias = Param(bias_tensor)
        new_layer = Linear(self._tail, weights, bias)
        # Todo: Will most likely need to rewrite the regularization layer implementation in the network
        if regularization:
            self._regularizations.append(Regularization(weights, self._loss, self._regularization_factor))

        self._add_layer(new_layer)

    def add_relu(self):
        new_layer = RelU(self._tail)
        self._add_layer(new_layer)

    def add_softmax(self):
        new_layer = SoftMax(self._tail)
        self._add_layer(new_layer)

    def inference(self, x):
        if self._forward is False:
            self._forward = True
        self._backward = False
        self._head.output = x
        self._head.forward()
        return self._tail.output

    def forward(self, x, y):
        output = self.inference(x)
        self._loss.actual = y
        self._loss.tail = self._tail
        self._loss.forward()
        self.loss = self._loss.output
        return output

    def accuracy(self, x, y):
        temp = self.forward(x,y)
        p = temp - temp.max(axis=0).values
        (p >= 0).to(torch.float64)
        return (p.argmax(axis=0) == y.argmax(axis=0)).to(torch.float64).mean()

    def backward(self):
        if self._forward:
            if not self._backward:
                # Setting up non-loss stuff
                # Managing the Regularization Layers
                for reg_lay in self._regularizations:
                    reg_lay.forward()
                self._objective_function = Sum(self._regularizations, self._loss)
                self._objective_function.forward()
                self._objective_function.output.backward()

                self._head.step(self._learning_rate)
                self._backward = True
                return True
            else:
                return False
        else:
            return False