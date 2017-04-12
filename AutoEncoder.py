import math, copy, random
import graphics as g
import numpy as np

def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoidPrime(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def relu(x):
    return max(0,x)

def reluPrime(x):
    return int(x > 0)

def printMatrix(m):
    m = m.value
    for i in range(len(m)):
        print(m[i].value)

class Neuron:
    def __init__(self):
        self.value = 0

    def activation(self, net):
        self.value = sigmoid(net)

    def __repr__(self):
        return string(self.value)

class Vector:
    def __init__(self, *args):
        self.value = list(args)

    def sigmoid(self):
        return Vector(*[sigmoid(i) for i in self.value])

    def apply(self, func):
        return Vector(*[func(i) for i in self.value])

    def set(self, key, value):
        self.value[key] = value
        return value

    def __imul__(self, other):
        t = type(other)
        if(t == Vector):
            for i in range(len(self.value)):
                self.value[i] *= other.value[i]
            return self
        elif(t == int or t == float):
            for i in range(len(self.value)):
                self.value[i] *= other
            return self

    def __mul__(self, other):
        t = type(other)
        j = copy.deepcopy(self)
        if((t == list and other[0] and (type(other[0]) == int or type(other[0]) == float)) or t == tuple):
            other = Vector(*other)
            t = type(other)
        elif(t == list and other[0] and (type(other[0]) == list or type(other[0]) == tuple)):  #return an array of vectors
            m = []
            for i in range(len(other)):
                # print(self * other[i])
                m.append(self * other[i])
            return m

        if(t == Vector):
            for i in range(len(j.value)):
                j.value[i] *= other.value[i]
            return j
        elif(t == int or t == float):
            for i in range(len(j.value)):
                j.value[i] *= other
            return j
        elif(t == Matrix):
            return other * self

    def __iadd__(self, other):
        t = type(other)
        if(t == Vector):
            for i in range(len(self.value)):
                self.value[i] += other.value[i]
            return self
        elif(t == int):
            for i in range(len(self.value)):
                self.value[i] += other
            return self

    def __add__(self, other):
        t = type(other)
        j = copy.deepcopy(self)
        if(t == Vector):
            for i in range(len(j.value)):
                j.value[i] += other.value[i]
            return j
        elif(t == int):
            for i in range(len(j.value)):
                j.value[i] += other
            return j

    def __isub__(self, other):
        t = type(other)
        if(t == Vector):
            for i in range(len(self.value)):
                self.value[i] -= other.value[i]
            return self
        elif(t == int):
            for i in range(len(self.value)):
                self.value[i] -= other
            return self

    def __sub__(self, other):
        t = type(other)
        j = copy.deepcopy(self)
        if(t == Vector):
            for i in range(len(j.value)):
                j.value[i] -= other.value[i]
            return j
        elif(t == int):
            for i in range(len(j.value)):
                j.value[i] -= other
            return j

    def __str__(self):
        return str(self.value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        return self.value[key]

class Matrix:
    def __init__(self, listOrxSize, ySize=None):
        t = type(listOrxSize)
        self.value = []
        if(t == list):
            t = type(listOrxSize[0])
            if(t == list or t == tuple):
                for i in range(len(listOrxSize)): #TODO: fix this up
                    self.value.append(Vector(*listOrxSize[i]))
            elif(t == Vector):
                self.value = listOrxSize
        elif(t == int):
            self.value = [Vector(*[0]*listOrxSize) for i in range(ySize)]

    def transpose(self):
        return Matrix(list(zip(*[x.value for x in self.value])))

    def sum(self):
        total = 0
        for v in self.value:
            total += sum(v)
        return total

    def apply(self, func):
        m = copy.deepcopy(self)
        for v in m.value:
            for i in range(len(v)):
                v.set(i, func(v[i]))
        return m

    def set(self, j, value):
        t = type(value)
        if(t == Vector):
            self.value[j] = value
        elif(t == int):
            self.value[j] = Vector(value)
        elif(t == list or t == tuple):
            self.value[j] = Vector(*value)

    def __len__(self):
        return len(self.value)

    def __str__(self):
        for i in range(len(self.value)):
            print(self.value[i])
        return ""

    def __add__(self, other):
        t = type(other)
        if(t == Matrix):
            m = copy.deepcopy(self)
            for i in range(len(self.value)):
                m.set(i, self[i]+other[i])
            return m
        elif(t == Vector):
            pass #TODO
        elif(t == int):
            m = copy.deepcopy(self)
            for i in range(len(m.value)):
                m.set(i, self[i]+other)
            return m
        elif(t == tuple or t == list):
            pass #TODO

    def __iadd__(self, other):
        t = type(other)
        if(t == Matrix):
            for i in range(len(self.value)):
                self.set(i, self[i]+other[i])
            return self
        elif(t == Vector):
            pass #TODO
        elif(t == int):
            pass #TODO
        elif(t == tuple or t == list):
            pass #TODO

    def __sub__(self, other):
        t = type(other)
        if(t == Vector):
            m = copy.deepcopy(self)
            for i in range(len(m.value)):
                m.set(i, m[i]-other[i])
            return m

    def __isub__(self, other):
        pass

    def __mul__(self, other):
        t = type(other)
        if(t == Vector):
            temp = []
            for i in self:
                temp.append(i.value)
            return Vector(*np.dot(temp,other)) #TODO, fix this (not always vector)
        elif(t == int):
            m = copy.deepcopy(self)
            for i in m.value:
                for j in range(len(i)):
                    i.set(j, i[j] * other)
            return m
        elif(t == Matrix):
            m = copy.deepcopy(self)
            for i in range(len(m.value)):
                m.set(i, m[i]*other[i])
            return m
    def __getitem__(self, key):
        return self.value[key]

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

class AutoEncoder:
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

target = [2.5,0.3]
nn = AutoEncoder(3,[2],2) #10 input, 2 hidden layer with 2 nodes, 10 output
nn.feedForward()

samples = 500
for i in range(samples):
    nn.backprop(target)
    nn.feedForward()
    if(not (i+1)%(samples/10)):
        print(i+1,"tries, error:",nn.cost(target))

screen = g.init(700,350, "NN.py")
while(g.hEvents()):
    g.begin()

    nn.draw(screen)

    g.end()
g.quit()
