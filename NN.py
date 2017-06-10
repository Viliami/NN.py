import numpy as np

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
        # return target-output
        return output-target
    return ((target - output) ** 2).mean(axis=0)/2

class Layer:
    def __init__(self, nodes, prevNodes):
        self.activation = np.random.random((nodes,1)) - 0.5
        self.netInput = np.random.random((nodes,1)) - 0.5
        self.bias = np.full((nodes,1), 0.1)

        #connecting this layer to the previous layer
        self.weights = np.random.random((nodes,prevNodes)) - 0.5

    def feedForward(self, prevLayer):
        self.netInput = np.dot(self.weights, prevLayer.activation) + self.bias
        self.activation = self.activationFunction(prevLayer.netInput)

    def backprop(self, error):
        return np.dot(nextLayer.weights.T, error[-1]) * self.activationFunction(layer.netInput, True)

    def backprop(self, target):
        target = np.array([target]).T
        costPrime = self.cost(self.layers[-1].activation,target,True)
        activationPrime = self.activationFunction(self.layers[-1].netInput,True)

        #maybe use temp np array instead of list
        error = [costPrime*activationPrime]
        for i in range(1,len(self.layers)):
            layer = self.layers[-1-i]
            nextLayer = self.layers[-i]
            error.append(np.dot(nextLayer.weights.T,error[-1]) * self.activationFunction(layer.netInput,True))
        return error

    def calcError(self): #calculate output layer error to backpropogate
        pass

class NN:
    def __init__(self, *layers):
        self.layers = [Layer(layers[0],0)]
        for i in range(1,len(layers)):
            self.layers.append(Layer(layers[i],layers[i-1]))
        self.__learningRate = 0.03 #arbitrary default learning rate
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
            print("set activation function to relu")
            self.__activationFunction = relu
        elif(value == "tanh"):
            print("tanh!1!")
            self.__activationFunction = tanh

    def feedForward(self, inputs):
        # if(len(inputs) != len(self.layers[0].activation)):
            # raise ValueError("Passed in incorrect amount of inputs")
        self.layers[0].activation = np.array([inputs],dtype=np.float64).T

        for i in range(1,len(self.layers)):
            layer = self.layers[i]
            layer.netInput = np.dot(layer.weights,self.layers[i-1].activation)+layer.bias
            layer.activation = self.activationFunction(layer.netInput)
        return self.layers[-1].activation

    def feedForwardM(self, input):
        self.layers[0].activation = input.T

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            layer.netInput = np.dot(layer.weights, self.layers[i-1].activation) + layer.bias
            layer.activation = self.activationFunction(layer.netInput)
        return self.layers[-1].activation[0]

    def backprop(self, target):
        target = np.array([target]).T
        costPrime = self.cost(self.layers[-1].activation,target,True)
        activationPrime = self.activationFunction(self.layers[-1].netInput,True)

        #maybe use temp array instead of list
        error = [costPrime*activationPrime]
        for i in range(1,len(self.layers)):
            layer = self.layers[-1-i]
            nextLayer = self.layers[-i]
            error.append(np.dot(nextLayer.weights.T,error[-1]) * self.activationFunction(layer.netInput,True))
        return error

    def backpropM(self, target):
        target = target.T
        costPrime = self.cost(self.layers[-1].activation, target, True)
        activationPrime = self.activationFunction(self.layers[-1].netInput, True)
        error = [costPrime * activationPrime]
        # print("error",error[-1].shape)

        for i in range(1, len(self.layers)):
            layer = self.layers[-1-i]
            nextLayer = self.layers[-i]
            error.append((nextLayer.weights.T @ error[-1]) * self.activationFunction(layer.netInput, True))

        return error

    def updateParamsM(self, error):

        for i in range(len(self.layers)-1):
            layer = self.layers[-1-i]
            layer.bias -= np.sum(error[i],axis=1).reshape(layer.bias.shape) * self.learningRate/error[i].shape[1]
            layer.weights -= np.dot(error[i],self.layers[-2-i].activation.T) * self.learningRate/error[i].shape[1]

    def updateParams(self, error): #stochastic gradient descent
        layerLen = len(self.layers)
        if(layerLen is not len(error)): #TODO:len-1
            raise ValueError("Incorrect error passed into sgd, not the same length as self.layers")

        for i in range(layerLen-1): #back to front
            layer = self.layers[-1-i]
            # layer.bias -= error[i] * (self.learningRate)
            layer.weights -= (error[i] * self.layers[-2-i].activation.T) * self.learningRate

    def gd(self, data): #gradient descent
        if(len(data.inputs) != len(data.outputs)):
            raise ValueError("Incorrect dataset input (inputs and outputs are different length)")

        self.feedForwardM(data.inputs)
        self.updateParamsM(self.backpropM(data.outputs))

    def train(self, data):
        for i in range(len(data.inputs)):
            self.feedForward(data.inputs[i])
            self.updateParams(self.backprop(data.outputs[i]))

    def minibatch(self, data, batchSize=10):
        size = len(data.inputs)
        # batch = np.random.permutation(size)
        for i in range(size//batchSize):
            # samples = batch[i*batchSize:(i+1)*batchSize]
            # self.feedForwardM(data.inputs[samples])
            self.feedForwardM(data.inputs[i*batchSize:(i+1)*batchSize])
            # self.updateParamsM(self.backpropM(data.outputs[samples]))
            self.updateParamsM(self.backpropM(data.outputs[i*batchSize:(i+1)*batchSize]))

    def evaluate(self, inputs, answers): #evaluate error on a dataset
        if(len(inputs) != len(answers)):
            raise ValueError("Incorrect dataset input (inputs and answers are different length)")
        count = 0
        # print(len(inputs))
        for i in range(len(inputs)):
            output = self.feedForward(inputs[i])
            # print(self.cost(output, answers[i]),output,answers[i])
            count += self.cost(output, answers[i])
        return (count/len(inputs))

    def test(self, data):
        inputs = data.inputs
        outputs = data.outputs
        numCorrect = 0
        for i in range(len(inputs)):
            output = self.feedForward(inputs[i]).reshape(10)
            if(np.argmax(outputs[i]) == np.argmax(output)):
                numCorrect += 1
        return numCorrect,len(inputs),str(numCorrect/len(inputs)*100)+"%"

    def __len__(self):
        return len(self.layers)

class CNN:
    def __init__(self):
        self.layers = []

    def feedForward(self):
        #for i in range(len(self.layers)):
        #    self.layers[i].feedForward()
        pass

    def backprop(self):
        pass

class ConvolutionalLayer:
    def __init__(self, nodes, prevNodes):
        self.activation = np.random.random((nodes,1)) - 0.5
        self.netInput = np.random.random((nodes,1)) - 0.5
        self.bias = np.full((nodes,1), 0.1)
        #connecting this layer to the previous layer
        self.weights = np.random.random((nodes,prevNodes)) - 0.5

    def convolution(self, a, b):
        pass

    def feedForward(self, prevLayer):
        # self.activationFunction()
        pass

    def backprop(self, error):
        pass

class PoolingLayer:
    def __init__(self):
        pass

    def feedForward(self):
        pass

    def backprop(self, error):
        pass
