# from graphics_qt import *
from graphics_pygame import *
from colour import Color
import numpy as np
import math

#TODO: create text labels for axis
#TODO: implement color gradients
#TODO: draw arrows

class Graph2D(Surface):
    def __init__(self, width, height, xAxis, yAxis):
        super().__init__(width, height)
        self.xAxis, self.yAxis = xAxis, yAxis
        self.showGrid = True
        self.points = []
        self.xLabel, self.yLabel = "x","y"
        self.xAxis = np.linspace(xAxis[0], xAxis[1], xAxis[2])
        self.yAxis = np.linspace(yAxis[0], yAxis[1], yAxis[2])

    def toPixel(self, cX, cY): #coordinates to pixels
        cX -= self.xAxis[0]
        cY -= self.yAxis[0]
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
        self.fill(self.backgroundColor)
        if(self.showGrid):
            self.renderGrid()
        for point in self.points:
            self.renderPoint(point)

    def renderLine(self, point, point2):
        self.line(self.toPixel(*point[0]), self.toPixel(*point2[0]), BLACK, 2)

    def renderPoint(self, point):
        self.filledCircle(self.toPixel(*point[0]), point[2], point[1], (0,0,0))

    def plot(self, x, y, color, radius):
        self.points.append(((x,y),color,radius))

    def plotFunction(self, func, color, accuracy=100):
        for x in self.xAxis:
            self.points.append((x,func(x)),color,radius)

class TimeSeries(Graph2D):
    def __init__(self, width, height, yAxis):
        super().__init__(width, height, (0,1,2), yAxis)
        #self.lines = 90
        self.POINT_LIMIT = 100
        self.offsetX = 0

    def setBackgroundColor(self, color):
        print(self.backgroundColor)
        self.backgroundColor = color

    def toPixel(self, cX, cY): #coordinates to pixels
        xRatio = self.width/len(self.xAxis)
        yRatio = self.height/len(self.yAxis)
        return (int(cX*xRatio), int(self.height - cY*yRatio ))

    def render(self):
        self.fill(self.backgroundColor)

        for i in range(1,len(self.points)):
            self.renderLine(self.points[i-1],self.points[i])

    def renderLine(self, point, point2):
        self.line(self.toPixel(point[0][0]-self.offsetX,point[0][1]), self.toPixel(point2[0][0]-self.offsetX,point2[0][1]), BLACK, 2)

    def plot(self, y):
        x = len(self.xAxis)+1
        if(len(self.points)+1 > self.POINT_LIMIT):
            self.points.pop(0)
            self.offsetX += 1
        self.xAxis = np.append(self.xAxis,len(self.xAxis)+1)
        self.points.append(((x,y), (255,0,0), 3))

class NeuralGrid(Graph2D): #only possible if there are at least 2 input neurons and 1 output
    def __init__(self, width, height, xAxis, yAxis, nn, data=None):
        super().__init__(width, height, xAxis, yAxis)
        self.nn = nn
        self.data = data
        coord = []
        for x in self.xAxis:
            for y in self.yAxis:
                coord.append(np.array([x,y]))
        self.coordinates = np.array(coord)
        print("CO",self.coordinates.shape)
        self.steps = 300
        self.colors = list(Color("red").range_to(Color("blue"),self.steps))

    def axisToPixel(self, cX, cY):
        cX -= self.xAxis[0]
        cY -= self.yAxis[0]
        w,h = self.width, self.height
        xRatio = w/len(self.xAxis)
        yRatio = h/len(self.yAxis)
        return (int(cX*xRatio), int(cY*yRatio))

    def render(self):
        self.fill(self.backgroundColor)
        a = self.nn.feedForward(self.coordinates).T
        counter = 0
        width = self.width//len(self.xAxis)*2
        amax = np.amax(a)
        if(amax < 1):
            amax = 1

        for y in self.yAxis:
        
            for x in self.xAxis:
                c = a[counter]*(((self.steps-1)//2)/amax)
                col = self.colors[int(c)].rgb
                self.filledSquare(self.toPixel(x,y),width,(col[0]*255,col[1]*255,col[2]*255),(col[0]*255,col[1]*255,col[2]*255))
                counter+=1

        if(self.data):
            for i in range(len(self.data.inputs)):
                x,y = self.data.inputs[i]
                c = self.data.outputs[i,0]*(((self.steps-1)//2)/amax)
                col = self.colors[int(c)].rgb
                self.renderPoint(((x,y),(col[0]*255,col[1]*255,col[2]*255),3))
                
    def generateColor(self): #TODO
        pass

class Grid(Graph2D):
    def __init__(self, width, height, xAxis, yAxis, data=None):
        super().__init__(width,height,xAxis,yAxis)
        self.data = data
        coord = []
        for x in self.xAxis:
            for y in self.yAxis:
                coord.append(np.array([x,y]))
        self.coordinates = np.array(coord)
        self.steps = 129
        self.colors = list(Color("red").range_to(Color("blue"),self.steps))
        self.offset = 70
        self.path = []
        self.points = [(150,110)]

    def toPixel(self, cX, cY): #coordinates to pixels
        cX -= self.xAxis[0]
        cY -= self.yAxis[0]
        xRatio = self.width/len(self.xAxis)
        yRatio = self.height/len(self.yAxis)
        return (int(cX*xRatio), int(cY*yRatio))

    def axisToPixel(self, cX, cY):
        offset = self.offset
        cX -= self.xAxis[0]
        cY -= self.yAxis[0]
        w,h = self.width-offset, self.height-offset
        xRatio = w/len(self.xAxis)
        yRatio = h/len(self.yAxis)
        return (int(cX*xRatio)+offset, int(cY*yRatio)+offset)

    def render(self):
        width = math.ceil(self.width/len(self.xAxis))
        self.fill(self.backgroundColor)
        a = self.data
        amax = np.amax(a)
        counter = 0
        for y in self.yAxis:
            for x in self.xAxis:
                c = a[int(x)][int(y)]*(((self.steps-1)//2)/amax)
                col = self.colors[int(c)].rgb
                self.filledSquare(self.axisToPixel(x,y),width,(col[0]*255,col[1]*255,col[2]*255),(col[0]*255,col[1]*255,col[2]*255))
                counter+=1

        #draw axis labels
        for y in self.yAxis:
            if(y%25 == 0):
                self.text(self.axisToPixel(-20,y-5),str(int(y)))
                self.line(self.axisToPixel(-5,y),self.axisToPixel(0,y))

        for x in self.xAxis:
            if(x%25 == 0):
                self.text(self.axisToPixel(x-5,-15),str(int(x)))
                self.line(self.axisToPixel(x,-5),self.axisToPixel(x,0))

        #draw border
        '''self.line((0,1),(self.width,1),width=3)
        self.line((self.width-1,0),(self.width-1,self.height),width=3)
        self.line((1,0),(1,self.height),width=3)
        self.line((0,self.height-1),(self.width,self.height-1),width=3)'''

        self.drawScatter(self.points)
        self.drawPath(self.path)

    def drawPath(self, path):
        prevPoint = path[0]
        circleColor = (100,255,0)
        lineColor = (0,0,0)
        color = (0,255,0)

        for i in range(len(path)-1):
            point = path[i]
            x,y = point
            x2,y2 = prevPoint
            self.line(self.axisToPixel(x,y),self.axisToPixel(x2,y2),lineColor,width=4)
            self.filledCircle(self.axisToPixel(x,y), 5, circleColor)
            prevPoint = point

        #draw arrowhead for last point
        height = 7
        width = 3.5
        lastPoint = np.array(path[-1])
        secondLastPoint = np.array(path[-2])

        direction = lastPoint-secondLastPoint
        direction /= np.linalg.norm(direction)

        q = np.array([direction[1],-direction[0]]) #rotate
        left = self.axisToPixel(*(lastPoint-(height*direction)+(width*q/2.0)))
        right = self.axisToPixel(*(lastPoint-(height*direction)-(width*q/2.0)))
        lastPoint = self.axisToPixel(*lastPoint)
        self.polygon([left,lastPoint,right])
        self.polygon([left,lastPoint,right],color,1)

    def drawScatter(self, points):
        for point in points:
            x,y = point
            self.filledCircle(self.axisToPixel(x,y),3)

class Structure(Surface):
    def __init__(self, width, height, nn):
        self.nn = nn
        super().__init__(width, height)

    def render(self): #TODO: add text value on hover
        self.fill(self.backgroundColor)
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
                radius = int(y_delta-(y_pad*2))
                radius = max(1, min(20, radius))
                screen.filledCircle((x, int(y)), radius, color)
                for k in range(len(layer.weights[j])):
                    prevLayer = self.nn.layers[i-1]
                    temp_y_delta = (h-(y_pad*2))/len(layer.weights[j])
                    weight = math.ceil(layer.weights[j,k])
                    wcolor = BLACK
                    if(weight <= 0):
                        wcolor = (200,0,0)
                        weight = -weight
                    screen.line((x,y), (x-x_delta,y_pad+(temp_y_delta//2)+int(temp_y_delta*k)), wcolor, weight)
                y += y_delta
            x += x_delta

class ImgArray(Surface):
    def __init__(self, width, height, array):
        super().__init__(width, height)
        self.array = array
        if(len(self.array.shape) == 1):
            self.array = self.array.reshape([int(np.sqrt(self.array.shape[0]))]*2)
        self.blittingSurface = pygame.surfarray.make_surface(self.array)

    def render(self):
        self.fill(self.backgroundColor)
        self.blit(self.blittingSurface, (0,0))

    def toRGB(self):
        print(self.array.shape)
        newArray = np.zeros((self.array.shape[0],self.array.shape[1], 3))
        y = 0
        for rows in self.array:
            x = 0
            for pixel in rows:
                newArray[x,y].fill(pixel)
                x += 1
            y += 1
        self.array = newArray
        self.blittingSurface = pygame.surfarray.make_surface(self.array)

    def toGrayScale(self):
        pass
