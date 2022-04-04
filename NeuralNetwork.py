class Network:
    def __init__(self, input_rows, output_rows, learning_rate=0.01, regularization_factor=0, loss='l2'):
        self._forward = False
        self._backward = False

        self._regularizations = []
        self._layers = []

        self._irows = input_rows
        self._orows = output_rows

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

    def add_linear(self, num_nodes, w=0.1, b=0, regularization=False):
        columns = self._tail.out_s
        rows = num_nodes
        weights = Param(torch.rand((rows, columns), device=DEVICE, requires_grad=True, dtype=torch.double))
        bias = Param(torch.rand((rows, 1), device=DEVICE, requires_grad=True, dtype=torch.double))
        with torch.no_grad():
            weights.output *= w
            bias.output *= b
        new_layer = Linear(self._tail, weights, bias)
        if regularization:
            self._regularizations.append(Regularization(weights, self._loss, self._regularization_factor))

        new_layer.tail = self._tail
        self._tail.head = new_layer
        self._tail = new_layer

    def add_relu(self):
        new_layer = RelU(self._tail)
        new_layer.tail = self._tail
        self._tail.head = new_layer
        self._tail = new_layer

    def add_softmax(self):
        new_layer = SoftMax(self._tail)

        new_layer.tail = self._tail
        self._tail.head = new_layer
        self._tail = new_layer

    def forward(self, x, y):
        if self._forward is False:
            self._forward = True
        self._backward = False
        self._head.output = x
        self._head.forward()
        self._loss.actual = y
        self._loss.tail = self._tail
        self._loss.forward()
        self.loss = self._loss.output
        return self._tail.output

    def accuracy(self, x, y):
        temp = self.forward(x,y)
        p = temp - temp.max(axis=0).values
        (p >= 0).to(DTYPE)
        return (p.argmax(axis=0) == y.argmax(axis=0)).to(DTYPE).mean()

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


class Layer:
    def __init__(self, tail):
        """
        :param output_shape (tuple): the shape of the output array.  When this isa single number, it gives the number of output neurons
            When this is an array, it gives the dimensions of the array of output neurons.
        """
        self.tail = tail
        self.head = None
        self.output = 0

    def forward(self):
        if self.head is not None:
            self.head.forward()

    def step(self, lr):
        if self.head is not None:
            self.head.step(lr)


class Input(Layer):
    def __init__(self, rows):
        super().__init__(None)
        self.tail = None
        self.out_s = rows
        self.output = 0

    def forward(self):
        super().forward()

    #def step(self, lr):
    #    super().step(lr)


class Param(Layer):
    def __init__(self, tensor):
        super().__init__(None)
        self.output = tensor

    def forward(self):
        """This layer's values do not change during forward propagation."""
        pass

    def step(self, lr):
        with torch.no_grad():
            self.output.data -= lr * (self.output.grad)
            self.output.grad.zero_()


class Linear(Layer):
    def __init__(self, x, W, b):
        """
        TODO: Accept any arguments specific to this child class.

        Raise an error if any of the argument's size do not match as you would expect.
        """
        super().__init__(x)
        self.W = W
        self.b = b
        self.out_s = self.b.output.shape[0]

    def forward(self):
        """
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = (self.W.output @ self.tail.output) + self.b.output
        super().forward()

    def step(self, lr):
        super().step(lr)
        self.W.step(lr)
        self.b.step(lr)


class RelU(Layer):
    def __init__(self, prev):
        super().__init__(prev)  # IDEK what to pass in here, not really needed.
        self.out_s = self.tail.out_s

    def forward(self):
        self.output = self.tail.output * (self.tail.output > 0)
        super().forward()

class SoftMax(Layer):
    def __init__(self, prev):
        """
        TODO: Accept any arguments specific to this child class.
        """
        super().__init__(prev)  # IDEK what to pass in here, not really needed.
        self.out_s = self.tail.out_s
        # Check if these are identical!

    def forward(self):
        self.output = torch.exp(self.tail.output) / torch.exp(self.tail.output).sum()
        super().forward()

class Sum(Layer):
    def __init__(self, *prevs):
        super().__init__(1)
        self.prevs = []
        for prev in prevs:
            if type(prev) is list or type(prev) is tuple:
                for prev2 in prev:
                    self.prevs.append(prev2)
            else:
                self.prevs.append(prev)

    def forward(self):
        temp = self.prevs[0].output.sum()
        for i in range(1, len(self.prevs)):
            temp = temp + self.prevs[i].output.sum()
        self.output = temp

    def step(self, lr):
        pass


class Regularization(Layer):
    def __init__(self, weight, loss, regularization_factor):
        super().__init__(weight)
        self.loss = loss
        self.regularization_factor = regularization_factor

    def forward(self):
        self.output = self.loss.output + (self.regularization_factor * ((self.tail.output ** 2).sum()))

    def step(self, lr):
        # Should not run?
        pass

class L2(Layer):
    def __init__(self):
        """
        TODO: Accept any arguments specific to this child class.
        """
        super().__init__(None)  # IDEK what to pass in here, not really needed.
        self.actual = 0
        # Check if these are identical!

    def forward(self):
        self.output = (self.actual - self.tail.output) ** 2
        self.output = self.output.sum(axis=0).mean()

    def step(self, lr):
        # Should be the end, therefor, step ends
        return

class CrossEntropy(Layer):
    def __init__(self):
        """
        TODO: Accept any arguments specific to this child class.
        """
        super().__init__(None)  # IDEK what to pass in here, not really needed.
        self.actual = 0
        # Check if these are identical!

    def forward(self):
        self.output = (self.actual*torch.log(self.tail.output + 1E-7))
        self.output = (self.output.sum(axis=0) * -1).mean()

    def step(self, lr):
        # Should be the end, therefor, step ends
        return