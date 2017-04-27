from mathlib import *
import graphics as g
import math, random

def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoidPrime(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def relu(x):
    return max(0,x)

def reluPrime(x):
    return int(x > 0)

# def rrelu(x):
#     self.u - self.l
#     return (x)
#
# def rreluPrime(x):
#     return int(x)

def linear(x):
    return x

def linearPrime(x):
    return 1

def elu(x): #TODO: accept a as hyperparameter
    a = 0.1
    if(x >= 0):
        return x
    return a*(math.exp(x)-1)

def eluPrime(x):
    a = 0.1
    if(x >= 0):
        return x
    return a*math.exp(x)

def tanh(x):
    return math.tanh(x)

def tanhPrime(x):
    return 1 - (math.tanh(x)**2)

def average(x):
    return float(sum(x))/max(len(x),1)

class Layer:
    def __init__(self, nodes, weights=None, bias=1):
        if(weights is None):
            weights = 0
        self.neurons = Vector(*[1 for i in range(nodes)])
        self.inputs = Vector(*[1 for i in range(nodes)])
        self.weights = Matrix(weights,nodes)
        variance = 0
        if weights != 0:
            variance = 1/weights
        print("variance",variance,"weights",weights)
        for i in range(nodes):
            # self.weights.set(i, Vector(*[random.random() for i in range(weights)]))
            self.weights.set(i, Vector(*[random.normalvariate(0,math.sqrt(variance)) for i in range(weights)]))
        # self.bias = Vector(*[random.random() for i in range(nodes)])
        self.bias = Vector(*[random.normalvariate(0,math.sqrt(variance)) for i in range(nodes)])

    def __str__(self):
        return str(self.neurons)

class NN:
    def __init__(self, inputNodes, hiddenLayers, outputNodes):
        self.layers = [Layer(inputNodes, 0)]
        t = [inputNodes,*hiddenLayers, outputNodes]
        for i in range(len(hiddenLayers)): #TODO: change this
            self.layers.append(Layer(hiddenLayers[i], t[i])) #num of nodes, num of prev nodes
        self.layers.append(Layer(outputNodes, t[-2]))
        self.error = 99999999
        self.learningRate = 0.4

    def setLearningRate(self, rate):
        self.learningRate = rate

    def setActivation(self, name, *params): #returns True if activation exists else False
        name = name.lower()
        if(name == "sigmoid"):
            self.activation = sigmoid
            self.activationPrime = sigmoidPrime
            return True
        elif(name == "relu"):
            self.activation = relu
            self.activationPrime = reluPrime
            return True
        elif(name == "elu"):
            self.activation = elu
            self.activationPrime = eluPrime
        elif(name == "tanh"):
            self.activation = tanh
            self.activationPrime = tanhPrime
        elif(name == "linear"):
            self.activation = linear
            self.activationPrime = linearPrime
        else:
            return False #TODO: throw exception

    def cost(self, target,output=None):
        if(not output):
            output = self.layers[-1].neurons
        t = type(target)
        if(type(target) == list):
            target = Vector(*target)
        self.error = sum((output-target).apply(lambda x:x**2))/2
        return self.error

    def costDerivative(self, target,output=None):
        if(not output):
            output = self.layers[-1].neurons
        t = type(target)
        if(type(target) == list):
            target = Vector(*target)
        return (output-target)

    def activation(self, x): #default is ReLU
        return relu(x)

    def activationPrime(self, x):
        return reluPrime(x)

    def setInput(self, inputValues):
        if(len(self.layers[0].neurons) == len(inputValues)):
            self.layers[0].neurons  = Vector(*inputValues)
            return True
        return False

    def predict(self, inputValues):
        if(self.setInput(inputValues)):
            return self.feedForward()
        else:
            return False

    def feedForward(self,inputValues=None):
        if(inputValues):
            self.setInput(inputValues)
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prevLayer = self.layers[i-1]
            for j in range(len(layer.neurons.value)):
                layer.inputs.set(j, sum(prevLayer.neurons*layer.weights[j]))
            layer.inputs += layer.bias
            layer.neurons = layer.inputs.apply(self.activation)
        return self.layers[-1].neurons

    def backprop(self, target, inputValues):
        self.feedForward(inputValues)
        if(type(target) == list):
            target = Vector(*target)
        output = self.layers[-1].neurons
        # error = [self.costDerivative(target,output) * output.apply(self.activationPrime)]
        error = [self.costDerivative(target,output) * self.layers[-1].inputs.apply(self.activationPrime)]

        for i in range(len(self.layers)-2, 0, -1):
            prevLayer = self.layers[i+1]
            currentLayer = self.layers[i]
            error.append((prevLayer.weights.transpose()*error[-1])*currentLayer.neurons.apply(self.activationPrime))
        return error

    def updateParams(self, error):
        for i in range(1,len(self.layers)):
            layer = self.layers[i]
            pLayer = self.layers[i-1] #previous layer
            err = error[-i]
            for j in range(len(layer.weights)):
                layer.bias.set(j, layer.bias[j] - self.learningRate * err[j])
                for k in range(len(pLayer.neurons)):
                    layer.weights[j].set(k, layer.weights[j][k] - self.learningRate * pLayer.neurons[k] * err[j])

    def train(self, data, targets, batchSize=1, debug=False): #NOTE: if batch does not evenly divide the data then data will be skipped
        size = len(data)
        if(batchSize <= 0):
            return
        if(batchSize == 1):
            for i in range(size):
                self.updateParams(self.backprop(targets[i], data[i]))
                if(not (i+1)%(size/10) and debug):
                    print(i+1,self.cost(targets[i]))
        else:
            err = []
            error = []
            c = 0
            for i in range(size):
                err.append(self.backprop(targets[i], data[i]))
                if(not (i+1)%batchSize):
                    c += 1
                    for j in range(len(err[0])):
                        t = 0
                        for k in range(len(err)):
                            t = err[k][j]+t
                        error.append(t/len(err))
                    self.updateParams(error)
                    if(not (i+1)%(size/10) and debug):
                        print(i+1,self.cost(targets[i]))
                    err = []
