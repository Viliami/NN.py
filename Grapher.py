# from graphics_qt import *
from graphics_pygame import *
from NN3 import *
import math

class Graph2D(Surface):
    def __init__(self, width, height, xAxis, yAxis):
        super().__init__(width, height)
        self.xAxis, self.yAxis = xAxis, yAxis
        self.showGrid = True
        self.points = []

    def toPixel(self, cX, cY): #coordinates to pixels
        xRatio = self.width/len(self.xAxis)
        yRatio = self.height/len(self.yAxis)
        return (int(cX*xRatio), int(cY*yRatio))

    def renderGrid(self):
        xPoints, yPoints = len(self.xAxis), len(self.yAxis)
        for x in range(0, xPoints+1):
            self.line(self.toPixel(x, 0), self.toPixel(x, self.height/yPoints))
        for y in range(0, yPoints+1):
            self.line(self.toPixel(0, y), self.toPixel(self.width/xPoints, y))

    def render(self):
        if(self.showGrid):
            self.renderGrid()
        for point in self.points:
            self.renderPoint(point)

    def renderPoint(self, point):
        self.filledCircle(self.toPixel(*point[0]), point[2], point[1], (0,0,0))

    def plot(self, x, y, color, radius):
        self.points.append(((x,y),color,radius))

class NeuralGrid(Graph2D): #only possible if there are at least 2 input neurons and 1 output
    def __init__(self, width, height, xAxis, yAxis, nn):
        super().__init__(width, height, xAxis, yAxis)
        self.nn = nn

    def render(self):
        # if(self.showGrid):
            # self.renderGrid()
        for x in self.xAxis:
            for y in self.yAxis:
                output = self.nn.feedForward(x,y)[0]
                c = min(255,max(0,output*255))
                # self.renderPoint(((x,y),(c,c,c),2))
                self.filledSquare(self.toPixel(x,y),11,(0,0,c))

        # if(data):
        #     for i in range(len(data.inputs)):
        #         x,y = data.inputs[i]
        #         c =  min(255,max(0,data.answers[i][0]*255))
        #         self.renderPoint(x,y,(0,c,c),True,5)

class Structure(Surface):
    def __init__(self, width, height, nn):
        self.nn = nn
        super().__init__(width, height)

    def render(self): #TODO: add text value on hover
        color=(66, 235, 244)
        screen = self
        w,h = self.width, self.height
        y_pad = 10
        x_pad = 10
        x_delta = (w-(2*x_pad))//len(self.nn.layers)
        x = x_delta//2
        layers_size = len(self.nn.layers)
        for i in range(layers_size):
            layer = self.nn.layers[i]
            y_delta = (h-(y_pad*2))/len(layer.activation)
            y = y_pad+(y_delta//2)
            for j in range(len(layer.activation)):
                ncolor = color
                if(layer.activation[j] <= 0):
                    ncolor = (200,0,0)
                radius = int(y_delta-(y_pad*2))
                radius = max(5, min(20, radius))
                screen.filledCircle((x, int(y)), radius, ncolor)
                for k in range(len(layer.weights[j])):
                    prevLayer = self.nn.layers[i-1]
                    temp_y_delta = (h-(y_pad*2))/len(layer.weights[j])
                    weight = math.ceil(layer.weights[j,k])
                    wcolor = BLACK
                    if(weight <= 0):
                        wcolor = (200,0,0)
                        weight = -weight
                    screen.line((x,y), (x-x_delta,y_pad+(temp_y_delta//2)+int(temp_y_delta*k)), wcolor, weight)
                y+=y_delta
            x += x_delta
        # screen.text((0,0),"Network structure")
