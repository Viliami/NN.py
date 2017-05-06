def relu(x, deriv=False):
    return x

def MSE(output, target, deriv=False):
    if(deriv):
        # return target-output
        return output-target
    return True

class Layer:
    def __init__(self, nodes, prevNodes):
        self.activation = np.random.random((nodes,1))
        self.netInput = np.random.random((nodes,1))
        self.bias = np.random.random((nodes,1))

        #connecting this layer to the previous layer
        self.weights = np.random.random((nodes,prevNodes))

class NN:
    def __init__(self, *layers):
        self.layers = [Layer(layers[0],0)]
        for i in range(1,len(layers)):
            self.layers.append(Layer(layers[i],layers[i-1]))
        self.__learningRate = 0.01 #arbitrary default learning rate
        self.__activationFunction = sigmoid
        self.cost = MSE

    @property
    def learningRate(self): return self.__learningRate

    @learningRate.setter
    def learningRate(self, value): self.__learningRate = value

    @property
    def activationFunction(self): return self.__activationFunction

    @activationFunction.setter
    def activationFunction(self, value):
        value = value.lower()
        if(value == "sigmoid"):
            self.__activationFunction = sigmoid
        elif(value == "relu"):
            self.__activationFunction = relu

    def feedForward(self, *inputs):
        if(len(inputs) != len(self.layers[0].activation)):
            raise ValueError("Passed in incorrect amount of inputs")
        self.layers[0].activation = np.array([inputs],dtype=np.float64).T

        for i in range(1,len(self.layers)):
            layer = self.layers[i]
            layer.netInput = np.dot(layer.weights,self.layers[i-1].activation)+layer.bias
            layer.activation = sigmoid(layer.netInput)
        return self.layers[-1].activation

    def backprop(self, *target):
        target = np.array([target]).T
        costPrime = self.cost(self.layers[-1].activation,target,True)
        activationPrime = self.activationFunction(self.layers[-1].netInput,True)

        #maybe use temp array instead of list
        error = [costPrime*activationPrime]
        for i in range(1,len(self.layers)):
            layer = self.layers[-1-i]
            nextLayer = self.layers[-i]
            error.append(np.dot(nextLayer.weights.T,error[-1]) * self.activationFunction(layer.netInput))

        return error

    def sgd(self, error): #stochastic gradient descent
        layerLen = len(self.layers)
        if(layerLen is not len(error)): #TODO:len-1
            raise ValueError("Incorrect error passed into sgd, not the same length as self.layers")

        for i in range(layerLen-1): #back to front
            layer = self.layers[-1-i]
            layer.bias -= error[i]
            layer.weights -= error[i]*self.layers[-2-i].activation.T

    def train(self, inputs, answers):
        if(len(inputs) is not len(answers)):
            raise ValueError("Incorrect dataset input (inputs and answers are different length)")
        for i in range(len(inputs)):
            self.feedForward(*inputs[i])
            nn.sgd(nn.backprop(*answers[i]))

    def evaluate(self, inputs, answers): #evaluate error on validation dataset
        pass

    def __len__(self):
        return len(self.layers)
