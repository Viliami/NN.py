import math, copy
import graphics as g

def sigmoid(x):
    return 1/(1+math.exp(-x))

def max(x):
    

class Vector:
    def __init__(self, *args):
        self.value = list(args)

    def sigmoid(self):
        return Vector(*[sigmoid(i) for i in self.value])

    def apply(self, func):
        return Vector(*[func(i) for i in self.value])

    def __imul__(self, other):
        t = other.__class__.__name__
        if(t == "Vector"):
            for i in range(len(self.value)):
                self.value[i] *= other.value[i]
            return self
        elif(t == "int"):
            for i in range(len(self.value)):
                self.value[i] *= other
            return self

    def __mul__(self, other):
        t = other.__class__.__name__
        j = copy.deepcopy(self)
        if(t == "Vector"):
            for i in range(len(self.value)):
                j.value[i] *= other.value[i]
            return j
        elif(t == "int"):
            for i in range(len(self.value)):
                j.value[i] *= other
            return j

    def __str__(self):
        return str(self.value)

    def __len__(self):
        return len(self.value)

class Neuron:
    def __init__(self):
        self.value = 0

    def activation(self, net):
        self.value = sigmoid(net)

    def __repr__(self):
        return string(self.value)

class Layer:
    def __init__(self, nodes):
        # self.neurons = [Neuron() for i in xrange(nodes)]
        self.neurons = Vector(*[0 for i in xrange(nodes)])
        self.weights = Vector(*[0 for i in xrange(nodes)])
        self.bias = Vector(*[0 for i in xrange(nodes)])

    def __str__(self):
        return str(self.neurons)

class AutoEncoder:
    def __init__(self, inputNodes, hiddenLayers, outputNodes):
        self.layers = [Layer(inputNodes)]
        for i in hiddenLayers:
            self.layers.append(Layer(i))
        self.layers.append(Layer(outputNodes))

    def cost(self, target):
        pass

    def feedForward(self):
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prevLayer = self.layers[i-1]
            # print(layer)
            # print(layer.weights)

    def draw(self, surface, color=(0,0,0)):
        w,h = surface.get_size()
        y_padding = 10
        for i in range(len(self.layers)):
            # print(len(self.layers[i].neurons))
            layer = self.layers[i]
            y = y_padding
            delta = (h-y_padding-50)/len(layer.neurons)
            print(delta)
            for j in range(len(layer.neurons)):
                y+=delta
                g.circle((20, y), 20, color)
        g.circle((w/2, h/2), 20, color)


nn = AutoEncoder(10,[2],10) #10 input, 1 hidden layer with 2 nodes, 10 output
nn.feedForward()

screen = g.init(450,450, "Ayy lmao")
while(g.hEvents()):
    g.begin()

    nn.draw(screen)

    g.end()
g.quit()
