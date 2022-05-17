from Layers import *

class Network:
    """
    A representation of a neural network using pytorch
    """
    def __init__(self, input_rows, output_rows, dtype=torch.float32, learning_rate=0.01, regularization_factor=0, loss='l2'):
        self._forward = False
        self._backward = False

        self._regularizations = []
        self._layers = []

        self._irows = input_rows
        self._orows = output_rows

        self._learning_rate = learning_rate
        self._regularization_factor = regularization_factor
        self._dtype = dtype

        self._head = Input(self._irows)
        self.objective_function = None
        if loss == 'l2':
            self._loss = L2()
        elif loss == 'cross entropy' or loss == 'ce' or loss == 'crossentropy':
            self._loss = CrossEntropy()
        else:
            raise ValueError("Get a actual loss you monke")
        self.loss = 0
        self.objective_function = Sum(self._loss)
        self._tail = self._head

    def _add_layer(self, new_layer):
        """
        Internal method for adding layer to the tail parameters
        :param new_layer:
        :return:
        """
        new_layer.tail = self._tail
        new_layer.head = self._loss
        self._loss.tail = new_layer
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
        weights = (torch.rand((rows, columns), dtype=self._dtype) + wo)*w
        bias = (torch.rand((rows, 1), dtype=self._dtype) + bo)*b
        return self.add_linear(weights, bias, True)

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
        self._add_layer(new_layer)
        if regularization:
            reg_lay = Regularization(weights, self._regularization_factor)
            self._regularizations.append(reg_lay)
            self.objective_function.add_reg(reg_lay)
            #self._add_layer(reg_lay)
            new_layer.reg = reg_lay
        return new_layer

    def add_relu(self):
        new_layer = RelU(self._tail)
        self._add_layer(new_layer)
        return new_layer

    def add_softmax(self):
        new_layer = SoftMax(self._tail)
        self._add_layer(new_layer)
        return new_layer

    def inference(self, x):
        self._head.output = x
        self._head.forward()
        return self._tail.output

    def forward(self, x, y):
        if self._forward is False:
            self._forward = True
        self._backward = False
        self._loss.actual = y
        output = self.inference(x)
        self.loss = float(self._loss.output)
        return output

    def accuracy(self, x, y):
        temp = self.forward(x,y)
        p = temp - temp.max(axis=0).values
        (p >= 0).to(self._dtype)
        return float((p.argmax(axis=0) == y.argmax(axis=0)).to(self._dtype).mean())

    def step(self):
        if self._backward:
            self._head.step(self._learning_rate)

    def zero_grad(self):
        self._head.zero_grad()

    def backward(self):
        if self._forward:
            if not self._backward:
                # Setting up non-loss stuff
                # Managing the Regularization Layers

                for reg_lay in self._regularizations:
                    reg_lay.head = self.objective_function
                    reg_lay.forward()
                self.objective_function.forward()
                for reg_lay in self._regularizations:
                    reg_lay.backward()
                self._loss.head = self.objective_function
                self._head.backward()
                # TODO: Regularization handling
                self._backward = True
                self._forward = False
                return True
            else:
                return False
        else:
            return False