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

class Layer:
    def __init__(self, nodes, weights=None, bias=1):
        if(weights is None):
            weights = 0
        self.neurons = Vector(*[1 for i in range(nodes)])
        self.weights = Matrix(weights,nodes)
        for i in range(nodes):
            self.weights.set(i, Vector(*[random.random() for i in range(weights)]))
        self.bias = Vector(*[random.random() for i in range(nodes)])

    def __str__(self):
        return str(self.neurons)

class NN:
    def __init__(self, inputNodes, hiddenLayers, outputNodes):
        self.layers = [Layer(inputNodes, 0)]
        t = [inputNodes,*hiddenLayers, outputNodes]
        for i in range(len(hiddenLayers)):
            self.layers.append(Layer(hiddenLayers[i], t[i])) #num of nodes, num of prev nodes
        self.layers.append(Layer(outputNodes, t[-2]))
        self.error = 99999999
        self.learningRate = 0.5

    def setActivation(self, name): #returns True if activation exists else False
        name = name.lower()
        if(name == "sigmoid"):
            self.activation = sigmoid
            self.activationPrime = sigmoidPrime
            return True
        elif(name == "relu"):
            self.activation = relu
            self.activationPrime = reluPrime
            return True
        else:
            return False

    def cost(self, target):
        output = self.layers[-1].neurons
        t = type(target)
        if(type(target) == list):
            target = Vector(*target)
        self.error = sum((target-output).apply(lambda x:x**2))/2
        return self.error

    def activation(self, x):
        # return sigmoid(x)
        return relu(x)

    def activationPrime(self, x): #currently sigmoid prime function
        # return sigmoidPrime(x)
        return reluPrime(x)

    def feedForward(self):
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prevLayer = self.layers[i-1]
            for j in range(len(layer.neurons.value)):
                layer.neurons.set(j, sum(prevLayer.neurons*layer.weights[j]))
            layer.neurons += layer.bias
            layer.neurons = layer.neurons.apply(self.activation)

    def backprop(self, target):
        if(type(target) == list):
            target = Vector(*target)
        output = self.layers[-1].neurons
        cost_derivative = (output - target)
        error = [cost_derivative * output.apply(sigmoidPrime)]

        for i in range(len(self.layers)-2, 0, -1):
            nextLayer = self.layers[i+1]
            currentLayer = self.layers[i]
            error.append((nextLayer.weights.transpose()*error[-1])*currentLayer.neurons.apply(self.activationPrime))

        for i in range(1,len(self.layers)):
            layer = self.layers[i]
            pLayer = self.layers[i-1] #previous layer
            err = error[-i]
            for j in range(len(layer.weights)):
                layer.bias.set(j, layer.bias[j] - self.learningRate * err[j])
                for k in range(len(pLayer.weights)):
                    layer.weights[j].set(k, layer.weights[j][k] - self.learningRate * pLayer.neurons[k] * err[j])

    def draw(self, surface, color=(0,0,0)):
        w,h = surface.get_size()
        y_pad = 10
        x_pad = 10
        x_delta = (w-(2*x_pad))/len(self.layers)
        x = x_delta/2
        layers_size = len(self.layers)
        for i in range(layers_size):
            layer = self.layers[i]
            y_delta = (h-(y_pad*2))/len(layer.neurons.value)
            y = y_pad+(y_delta/2)
            for j in range(len(layer.neurons.value)):
                for k in range(len(layer.weights[j])):
                    prevLayer = self.layers[i-1]
                    temp_y_delta = (h-(y_pad*2))/len(layer.weights[j])
                    g.line((x,y), (x-x_delta,y_pad+(temp_y_delta/2)+(temp_y_delta*k)), color)
                g.circle((x, y), min(20, y_delta-(y_pad*2)), color)
                y+=y_delta
            x += x_delta
