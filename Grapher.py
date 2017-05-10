# from graphics_qt import *
from graphics_pygame import *
from NN3 import *

class Graph2D(Surface):
    def __init__(self, width, height, xAxis, yAxis):
        super().__init__(width, height)
        print(self.width,self.height)
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
            self.line(self.toPixel(x, 0), self.toPixel(x, self.height))
        for y in range(0, yPoints+1):
            self.line(self.toPixel(0, y), self.toPixel(self.width, y))

    def render(self):
        self.renderGrid()
        for point in self.points:
            self.filledCircle(self.toPixel(*point[0]), point[2], point[1], (255,0,0))

    def plot(self, x, y, color, radius):
        self.points.append(((x,y),color,radius))
