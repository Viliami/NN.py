import math, copy, random
import graphics as g

def sigmoid(x):
    return 1/(1+math.exp(-x))

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
            for i in range(len(j.value)):
                j.value[i] *= other.value[i]
            return j
        elif(t == "int"):
            for i in range(len(j.value)):
                j.value[i] *= other
            return j

    def __iadd__(self, other):
        t = other.__class__.__name__
        if(t == "Vector"):
            for i in range(len(self.value)):
                self.value[i] += other.value[i]
            return self
        elif(t == "int"):
            for i in range(len(self.value)):
                self.value[i] += other
            return self

    def __add__(self, other):
        t = other.__class__.__name__
        j = copy.deepcopy(self)
        if(t == "Vector"):
            for i in range(len(j.value)):
                j.value[i] += other.value[i]
            return j
        elif(t == "int"):
            for i in range(len(j.value)):
                j.value[i] += other
            return j

    def __str__(self):
        return str(self.value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        return self.value[key]

class Layer:
    def __init__(self, nodes, weights=None, bias=1):
        if(weights is None):
            weights = 0
        self.neurons = Vector(*[1 for i in range(nodes)])
        self.weights = []
        for i in range(nodes):
            self.weights.append(Vector(*[random.random() for i in range(weights)]))
        self.bias = Vector(*[1 for i in range(nodes)])

    def __str__(self):
        return str(self.neurons)

class AutoEncoder:
    def __init__(self, inputNodes, hiddenLayers, outputNodes):
        self.layers = [Layer(inputNodes, 0)]
        t = [inputNodes,*hiddenLayers, outputNodes]
        for i in range(len(hiddenLayers)):
            self.layers.append(Layer(hiddenLayers[i], t[i])) #num of nodes, num of prev nodes
        self.layers.append(Layer(outputNodes, t[-2]))

    def cost(self, target):
        pass

    def feedForward(self):
        print(self.layers[0].neurons)
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prevLayer = self.layers[i-1]
            for j in range(len(layer.neurons)):
                layer.neurons.set(j, sum((prevLayer.neurons*layer.weights[j]).value))
            layer.neurons = layer.neurons.apply(sigmoid)
            print(layer.neurons)
            layer.neurons += layer.bias
            print(layer.neurons)

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
                for k in range(len(layer.weights[j])):
                    prevLayer = self.layers[i-1]
                    temp_y_delta = (h-(y_pad*2))/len(layer.weights[j])
                    g.line((x,y), (x-x_delta,y_pad+(temp_y_delta/2)+(temp_y_delta*k)), color)
                g.circle((x, y), min(20, y_delta-(y_pad*2)), color)
                y+=y_delta
            x += x_delta

nn = AutoEncoder(3,[5,5],3) #10 input, 1 hidden layer with 2 nodes, 10 output
nn.feedForward()

screen = g.init(700,350, "NN.py")
while(g.hEvents()):
    g.begin()

    nn.draw(screen)

    g.end()
g.quit()
