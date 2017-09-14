import numpy as np

#TODO: confusion matrix

# np.random.seed(1)
def sigmoid(x,deriv=False):
    if(deriv):
        sig = sigmoid(x)
        return sig*(1 - sig)
    return 1/(1+np.exp(-x))

def relu(x, deriv=False):
    if(deriv):
        y = np.array(x)
        y[x > 0] = 1
        y[x < 0] = 0
        return y
    return x * (x > 0)

def tanh(x, deriv=False):
    if(deriv):
        return (np.tanh(x)*np.tanh(x))*-1 + 1
    return np.tanh(x)

def MSE(output, target, deriv=False):
    if(deriv):
        return output-target
    return ((target - output) ** 2).mean(axis=0)/2

class NN:
    def __init__(self, *layers):
        self.nlayers = layers
        self.layers = [Layer(layers[0],0)]
        for i in range(1,len(layers)):
            self.layers.append(Layer(layers[i],layers[i-1]))
        self.__learningRate = 0.03 #arbitrary default learning rate
        self.__activationFunction = sigmoid
        self.cost = MSE
        self.momentum = 0
        self.prevUpdate = [[0,0] for layer in self.layers]

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
        elif(value == "tanh"):
            self.__activationFunction = tanh
        for layer in self.layers:
            layer._activationFunction = self.__activationFunction

    def updateParams(self, error):
        for i in range(len(self.layers)-1):
            layer = self.layers[-1-i]
            a = np.sum(error[i],axis=1).reshape(layer.bias.shape) * self.learningRate/error[i].shape[1]
            b = np.dot(error[i],self.layers[-2-i].activation.T) * self.learningRate/error[i].shape[1]
            layer.bias -= a + (self.momentum*self.prevUpdate[i][0])
            layer.weights -= b + (self.momentum*self.prevUpdate[i][1])
            self.prevUpdate[i] = [a,b]

    def backprop(self, target):
        error = [self.layers[-1].calcError(target)]

        for i in range(1, len(self.layers)):
            layer = self.layers[-1-i]
            nextLayer = self.layers[-i]
            error.append(layer.backprop(error[-1], nextLayer))

        return error

    def feedForward(self, inputs): #TODO: incorrect output for 1d input
        self.layers[0].activation = inputs.T

        for i in range(1, len(self.layers)):
            self.layers[i].feedForward(self.layers[i-1])

        return self.layers[-1].activation

    def train(self, data):
        self.updateParams(self.backprop(data.outputs))
        '''for i in range(len(data.inputs)):
            self.feedForward(data.inputs[i])
            self.updateParams(self.backprop(data.outputs[i]))'''

    def minibatch(self, data, batchSize=10,momentum=False): #minibatch training on dataset sample
        size = len(data.inputs)
        if(size//batchSize == 0):
            raise ValueError("Batch size can't be greater than dataset size")
        for i in range(size//batchSize):
            index = i*batchSize
            self.feedForward(data.inputs[index:index+batchSize])
            self.updateParams(self.backprop(data.outputs[index:index+batchSize]))

    def batchTrain(self, data): #batch training on full dataset
        self.feedForward(data.inputs)
        self.updateParams(self.backprop(data.outputs))

    def evaluate(self, dataset): #evaluate error on a dataset

        if(len(dataset.inputs) != len(dataset.outputs)):
            raise ValueError("Incorrect dataset input (inputs and outputs are different length)")

        output = self.feedForward(dataset.inputs).T
        return self.cost(output,dataset.outputs)

    def test(self, data):
        inputs = data.inputs
        outputs = data.outputs
        numCorrect = 0
        for i in range(len(inputs)):
            output = self.feedForward(inputs[i]).reshape(10)
            if(np.argmax(outputs[i]) == np.argmax(output)):
                numCorrect += 1
        return numCorrect,len(inputs),str(numCorrect/len(inputs)*100)+"%"

    def createCopy(self):
        copy = self.__class__(*self.nlayers)
        copy.learningRate = self.learningRate
        copy.__activationFunction = self.__activationFunction
        copy.momentum = self.momentum
        for i in range(len(copy.layers)):
            np.copyto(copy.layers[i].weights,self.layers[i].weights)
            np.copyto(copy.layers[i].bias,self.layers[i].bias)
            copy.layers[i]._activationFunction = self.layers[i]._activationFunction
        return copy

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def prune(self, test):
        pass

    def __len__(self):
        return len(self.layers)

class Layer:
    def __init__(self, nodes, prevNodes):
        self.activation = np.random.random((nodes,1)) - 0.5
        self.netInput = np.random.random((nodes,1)) - 0.5
        self.bias = np.full((nodes,1), 0.1)
        self._cost = MSE
        self._activationFunction = relu

        #connecting this layer to the previous layer
        self.weights = np.random.random((nodes,prevNodes)) - 0.5

    def feedForward(self, prevLayer):
        self.netInput = np.dot(self.weights, prevLayer.activation) + self.bias
        self.activation = self.activationFunction() #TODO: rewrite activation member var out

    def backprop(self, error, nextLayer):
        return np.dot(nextLayer.weights.T, error) * self.activationFunction(True)

    def calcError(self, target): #calculate output layer error to backpropogate
        costPrime = self.cost(target.T, True)
        activationPrime = self.activationFunction(True)
        return costPrime * activationPrime

    def cost(self, target, deriv=False):
        return self._cost(self.activation, target, deriv)

    def activationFunction(self, deriv=False):
        return self._activationFunction(self.netInput, deriv)

class CNN:
    def __init__(self):
        self.layers = [ConvolutionalLayer((28,28), (5,5))]

    def addLayer(self, layer):
        self.layers.append(layer)

    def feedForward(self, inputs):
        self.layers[0].map = inputs
        for i in range(1,len(self.layers)):
            self.layers[i].feedForward(self.layers[i-1]) 

class ConvolutionalLayer: #2d matrix as input, 2d matrix as output
    def __init__(self, size, filterSize):
        self.map = None
        self.filter = None

    def convolution(self, a, b):
        return signal.fftconvolve(a,b,mode="valid")

    def feedForward(self, prevLayer):
        #TODO: handle multiple maps in a single layer
        self.map = self.convolution(prevLayer.map, self.filter)

    def backprop(self, error, nextLayer):
        pass

class FlatteningLayer:
    def __init__(self):
        pass

    def feedForward(self):
        pass

    def backprop(self, error, nextLayer):
        pass

class PoolingLayer:
    def __init__(self):
        pass

    def feedForward(self):
        pass

    def backprop(self, error):
        pass

class RELULayer:
    def __init__(self):
        pass
