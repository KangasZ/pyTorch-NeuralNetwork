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

    def backward(self):
        if self.head is not None:
            self.head.backward()
        else:
            self.gradient = self.output

    def step(self, lr):
        if self.head is not None:
            self.head.step(lr)

    def zero_grad(self):
        self.gradient = 0
        if self.head is not None:
            self.head.zero_grad()


class Input(Layer):
    def __init__(self, rows):
        super().__init__(None)
        self.tail = None
        self.out_s = rows
        self.output = 0


class Param(Layer):
    def __init__(self, tensor):
        super().__init__(None)
        self.output = tensor

    def forward(self):
        """This layer's values do not change during forward propagation."""
        return

    def backward(self):
        print("this shouldnt be called bud")
        return

    def step(self, lr):
        self.output -= self.gradient*lr
        #print(self.gradient)


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

    def backward(self):
        super().backward()
        #print(self.head.gradient)
        self.gradient = torch.matmul(torch.t(self.W.output), self.head.gradient)
        self.W.gradient += torch.matmul(self.head.gradient, torch.t(self.tail.output))
        #self.W.backward(lr)
        self.b.gradient = self.head.gradient
        #self.b.backward(lr)

    def step(self, lr):
        super().step(lr)
        self.W.step(lr)
        self.b.step(lr)

    def zero_grad(self):
        super().zero_grad()
        self.W.zero_grad()
        self.b.zero_grad()


class RelU(Layer):
    def __init__(self, prev):
        super().__init__(prev)  # IDEK what to pass in here, not really needed.
        self.out_s = self.tail.out_s

    def forward(self):
        self.output = self.tail.output * (self.tail.output > 0)
        super().forward()

    def backward(self):
        super().backward()
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
        self.output = torch.div(torch.exp(self.tail.output), torch.exp(self.tail.output).sum(axis=0))
        super().forward()

    def backward(self):
        # adapted from https://e2eml.school/softmax.html
        super().backward()
        assert type(self.head is CrossEntropy)
        #self.gradient = (self.output * (torch.eye(self.output.shape[0]) - (self.output**2).sum()
        #                        ))\
        #                @ self.head.gradient
        self.gradient = self.head.gradient


class Sum(Layer):
    def __init__(self, prev):
        super().__init__(1)
        self.prevs = []
        self.prevs.append(prev)

    def add_reg(self, prev):
        self.prevs.append(prev)
        #prev.sum = self

    def forward(self):
        temp = self.prevs[0].output.sum()
        for i in range(1, len(self.prevs)):
            temp += self.prevs[i].output.sum()
        self.output = temp
        super().forward()

    def backward(self):
        # Todo: If sum is used anywhere beside the end it breaks. Dont have that happen.
        #super().step(lr)
        self.gradient = self.output


class Regularization(Layer):
    def __init__(self, weight, regularization_factor):
        super().__init__(weight)
        self.regularization_factor = regularization_factor

    def forward(self):
        self.output = (self.regularization_factor * ((self.tail.output ** 2).sum()))

    def backward(self):
        self.gradient = self.regularization_factor*2*self.tail.output
        self.tail.gradient += self.gradient

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

    def backward(self):
        super().backward()
        self.gradient = 2*(self.actual - self.tail.output)*-1
        #print("L2 Grad:", self.gradient)

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

    def backward(self):
        # Calculated on my own similar to l2 layer
        super().backward()
        #self.gradient = ((1/self.tail.output)*self.output*torch.div(self.actual, self.output) * -1).mean(axis=1)
        #self.gradient = self.gradient.reshape(self.gradient.shape[0], 1)
        self.gradient = (self.tail.output - self.actual) * self.head.gradient
