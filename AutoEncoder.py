import math, copy, random
import graphics as g

def sigmoid(x):
    return 1/(1+math.exp(-x))

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

    def set(self, j, value):
        self.value[j] = value

    def __str__(self):
        for i in range(len(self.value)):
            print(self.value[i])
        return ""

    def __mul__(self, other):
        t = type(other)
        if(t == Vector):
            m = Matrix(len(self.value[0]),len(self.value))
            for i in range(len(self.value)):
                m.set(i, self[i] * other)
            return m

    def __getitem__(self, key):
        return self.value[key]

class Layer:
    def __init__(self, nodes, weights=None, bias=1):
        if(weights is None):
            weights = 0
        self.neurons = Vector(*[1 for i in range(nodes)])
        self.neurons = Matrix(1,nodes)
        print("n",self.neurons)
        self.weights = Matrix(weights,nodes)
        for i in range(nodes):
            self.weights.set(i, Vector(*[random.random() for i in range(weights)]))
        self.bias = Vector(*[random.random() for i in range(nodes)])

    def __str__(self):
        return str(self.neurons)

print("matrix",Matrix(1,5))

class AutoEncoder:
    def __init__(self, inputNodes, hiddenLayers, outputNodes):
        self.layers = [Layer(inputNodes, 0)]
        t = [inputNodes,*hiddenLayers, outputNodes]
        for i in range(len(hiddenLayers)):
            self.layers.append(Layer(hiddenLayers[i], t[i])) #num of nodes, num of prev nodes
        self.layers.append(Layer(outputNodes, t[-2]))
        self.error = 99999999

    def cost(self, target):
        output = self.layers[-1].neurons
        t = type(target)
        if(type(target) == list):
            target = Vector(*target)
        self.error = sum((target-output).apply(lambda x:x**2))/2
        return self.error

    def feedForward(self):
        # print(self.layers[0].neurons)
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prevLayer = self.layers[i-1]
            for j in range(len(layer.neurons)):
                layer.neurons.set(j, sum((prevLayer.neurons*layer.weights[j]).value))
            layer.neurons += layer.bias
            layer.neurons = layer.neurons.apply(sigmoid)

    def backprop(self, target):
        if(type(target) == list):
            target = Vector(*target)
        output = self.layers[-1].neurons
        out_sig = output.sigmoid()
        output_error = out_sig * ((out_sig*-1) + 1)
        cost_derivative = (output - target)
        err = (output - target) * output_error
        print("out:",output)
        print("target:",target)
        print(err)
        #output layer done

        currentLayer = self.layers[-2]
        prevLayer = self.layers[-1]
        w = prevLayer.weights
        print("w",w)
        t = prevLayer.weights.transpose()
        print("t",t)
        print("err",err)
        print("err*t",err*t)
        print(t)
        print("neurons")
        print(currentLayer.neurons)

        # for i in range(len(self.layers)-1, 0, -1):
        #     print(i)

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
nn.backprop([1,2,2])

screen = g.init(700,350, "NN.py")
while(g.hEvents()):
    g.begin()

    nn.draw(screen)

    g.end()
g.quit()
