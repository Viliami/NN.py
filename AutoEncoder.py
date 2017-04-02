import math, copy
import graphics as g

def sigmoid(x):
    return 1/(1+math.exp(-x))

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
    def __init__(self, nodes, weights=None):
        if(weights is None):
            weights = nodes
        # self.neurons = [Neuron() for i in range(nodes)]
        self.neurons = Vector(*[0 for i in range(nodes)])
        self.weights = Vector(*[0 for i in range(weights)])
        print(self.weights)
        self.bias = Vector(*[0 for i in range(nodes)])

    def __str__(self):
        return str(self.neurons)

class AutoEncoder:
    def __init__(self, inputNodes, hiddenLayers, outputNodes):
        self.layers = [Layer(inputNodes, 0)]
        t = [inputNodes,*hiddenLayers, outputNodes]
        for i in range(len(hiddenLayers)):
            self.layers.append(Layer(hiddenLayers[i], t[i]))
        self.layers.append(Layer(outputNodes, t[-2]))

    def cost(self, target):
        pass

    def feedForward(self):
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prevLayer = self.layers[i-1]

    def draw(self, surface, color=(0,0,0)):
        w,h = surface.get_size()
        y_pad = 10
        x_pad = 10
        x_delta = (w-(2*x_pad))/len(self.layers)
        x = x_delta/2
        layers_size = len(self.layers)
        for i in range(layers_size):
            layer = self.layers[i]
            y_delta = (h-(y_pad*2))/len(layer.neurons)
            y = y_pad+(y_delta/2)
            for j in range(len(layer.neurons)):
                for k in range(len(layer.weights)):
                    prevLayer = self.layers[i-1]
                    temp_y_delta = (h-(y_pad*2))/len(layer.weights)
                    g.line((x,y), (x-x_delta,y_pad+(temp_y_delta/2)+(temp_y_delta*k)), color)
                g.circle((x, y), min(20, y_delta-(y_pad*2)), color)
                y+=y_delta
            x += x_delta


nn = AutoEncoder(4,[3,2,3],4) #10 input, 1 hidden layer with 2 nodes, 10 output
nn.feedForward()

screen = g.init(750,750, "NN.py")
while(g.hEvents()):
    g.begin()

    nn.draw(screen)

    g.end()
g.quit()
