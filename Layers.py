import torch

class Layer:
    def __init__(self, tail):
        """
        :param output_shape (tuple): the shape of the output array.  When this isa single number, it gives the number of output neurons
            When this is an array, it gives the dimensions of the array of output neurons.
        """
        self.tail = tail
        self.head = None
        self.output = 0
        self.gradient = 0

    def forward(self):
        if self.head is not None:
            self.head.forward()

    def step(self, lr):
        if self.head is not None:
            self.head.step(lr)
            print(self.head.gradient)
        else:
            self.gradient = self.output


class Input(Layer):
    def __init__(self, rows):
        super().__init__(None)
        self.tail = None
        self.out_s = rows
        self.output = 0

    def forward(self):
        super().forward()

    def step(self, lr):
        super().step(lr)


class Param(Layer):
    def __init__(self, tensor):
        super().__init__(None)
        self.output = tensor

    def forward(self):
        """This layer's values do not change during forward propagation."""
        return

    def step(self, lr):
        print("Hey how about you stop using this and compute your own derivative, huh?")
        print(self.output)
        print(self.gradient)
        self.output -= self.gradient*lr



class Linear(Layer):
    def __init__(self, x, W, b):
        """
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
        print(self.head.gradient)
        # Todo: Fix this gradient
        self.gradient = self.head.gradient
        self.W.gradient = torch.matmul(self.head.gradient, torch.t(self.tail.output))
        self.W.step(lr)
        self.b.gradient = self.output
        self.b.step(lr)


class RelU(Layer):
    def __init__(self, prev):
        super().__init__(prev)  # IDEK what to pass in here, not really needed.
        self.out_s = self.tail.out_s

    def forward(self):
        self.output = self.tail.output * (self.tail.output > 0)
        super().forward()

    def step(self, lr):
        super().step(lr)
        # Todo: Make work with SGD
        # My thought is mean along row then round to 0 or 1
        self.gradient = self.head.gradient * (self.output > 0)


class SoftMax(Layer):
    def __init__(self, prev):
        """
        """
        super().__init__(prev)  # IDEK what to pass in here, not really needed.
        self.out_s = self.tail.out_s
        # Check if these are identical!

    def forward(self):
        self.output = torch.exp(self.tail.output) / torch.exp(self.tail.output).sum()
        super().forward()

    def step(self, lr):
        super().step(lr)
        # todo: fix
        self.gradient = self.head.gradient*1

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
        # Todo: If sum is used anywhere beside the end it breaks. Dont have that happen.
        #super().step(lr)
        self.gradient = self.output


class Regularization(Layer):
    def __init__(self, weight, loss, regularization_factor):
        super().__init__(weight)
        self.loss = loss
        self.regularization_factor = regularization_factor

    def forward(self):
        self.output = self.loss.output + (self.regularization_factor * ((self.tail.output ** 2).sum()))

    def step(self, lr):
        super().step(lr)
        self.gradient = self.head.gradient*1


class L2(Layer):
    def __init__(self):
        """
        TODO: Accept any arguments specific to this child class.
        """
        super().__init__(None)  # IDEK what to pass in here, not really needed.
        self.actual = 0
        self.intermediate = 0
        # Check if these are identical!

    def forward(self):
        self.intermediate = (self.actual - self.tail.output) ** 2
        self.output = self.intermediate.sum(axis=0).mean()

    def step(self, lr):
        super().step(lr)
        print("Intermediate L2", self.intermediate)
        print("Tail grad:", self.head.gradient)
        self.gradient = self.head.gradient * self.intermediate
        print("L2 Grad:", self.gradient)

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
        self.step(lr)
        self.gradient = self.head.gradient*1