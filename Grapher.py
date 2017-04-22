import pygame, pygame.gfxdraw

BLACK = (0,0,0)
WHITE = (255,255,255)

#TODO: change from pygame to graphics

class Grapher:
    def __init__(self, screen, gridWidth=10, gridHeight=10, gridShown=False, radius=4):
        self.screen = screen
        self.width, self.height =  screen.getSize()
        self.radius = radius
        self.gridShown = gridShown
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.points = []
        self.lines = []

    def showGrid(self):
        self.gridShown = True
    def hideGrid(self):
        self.gridShown = False

    def setGridColor(self, color):
        self.gridColor = BLACK

    def setBackgroundColor(self, color):
        self.backgroundColor = WHITE

    def changeWidth(width):
        # self.screen.surface = pygame.transform.scale(self.screen,(width,self.height))
        pass

    def setHeight(height):
        # self.screen = pygame.transform.scale(self.screen,(self.width,height))
        pass

    def linePoint(self, m, c, x): #gets y at a point x
        return m*x + c

    def surfaceToGraph(self, pos):
        x,y = pos
        y = (self.height-y)
        gWidth = float(self.width)/self.gridWidth
        gHeight = float(self.height)/self.gridHeight
        return (x/gWidth,y/gHeight)

    def graphToSurface(self, pos):
        x,y = pos
        y = (self.gridHeight-y)
        gWidth = float(self.width)/self.gridWidth
        gHeight = float(self.height)/self.gridHeight
        return (x*gWidth,y*gHeight)

    def plot(self, x, y, color=BLACK):
        # save plots to array
        x,y = self.graphToSurface((x,y))
        self.points.append((x,y,color))

    def plotLine(self, startPosOrM, endPosOrC, color=BLACK):
        if(type(startPosOrM) is list):
            sPos = self.graphToSurface(startPosOrM)
            ePos = self.graphToSurface(endPosorC)
            self.lines.append((sPos, ePos, color))
        elif(type(startPosOrM) is int or type(startPosOrM) is float):
            m, c = (startPosOrM, endPosOrC)
            sPos = self.graphToSurface((0,c))
            ePos = self.graphToSurface((self.gridWidth, self.gridWidth*m + c))
            self.lines.append((sPos, ePos, color))

    def plotCircle(self, center, radius, color=BLACK):
        sPos = self.graphToSurface(center)
        if(self.width == self.height):
            radius *= float(self.width)/self.gridWidth
            pygame.gfxdraw.aacircle(self.surface, int(sPos[0]), int(sPos[1]), int(radius), color)

    def fillSquare(self, x,y, color=BLACK):
        x, y = self.graphToSurface((x,y))
        gWidth = float(self.width)/self.gridWidth
        gHeight = float(self.height)/self.gridHeight
        pygame.draw.rect(self.surface, color, [x,y,gWidth,gHeight])

    def plotFilledCircle(self, center, radius, color=BLACK):
        sPos = self.graphToSurface(center)
        if(self.width == self.height):
            radius *= float(self.width)/self.gridWidth
            pygame.gfxdraw.aacircle(self.surface, int(sPos[0]), int(sPos[1]), int(radius), color)
            pygame.gfxdraw.filled_circle(self.surface, int(sPos[0]), int(sPos[1]), int(radius), color)

    def renderPoint(self, x, y, color):
        if(self.radius > 1):
            pygame.gfxdraw.aacircle(self.surface, int(x),int(y),self.radius, (color[0],color[1],color[2]))
            pygame.gfxdraw.filled_circle(self.surface, int(x),int(y),self.radius, color)
        else:
            self.surface.fill(color, ((x,y),(2,2)))

    def renderLine(self, startPos, endPos, color):
        pygame.draw.aaline(self.surface, color, startPos, endPos)

    def renderGrid(self):
        gWidth=float(self.width)/self.gridWidth
        gHeight=float(self.height)/self.gridHeight
        for x in range(1,self.gridWidth):
            pygame.draw.line(self.surface, BLACK, (x*gWidth, 0),(x*gWidth, self.height))
        for y in range(1, self.gridHeight):
            pygame.draw.line(self.surface, BLACK, (0, y*gHeight),(self.width, y*gHeight))

    def render(self):
        if(self.gridShown):
            self.renderGrid()

        for point in self.points:
            self.renderPoint(point[0], point[1], point[2])
        for line in self.lines:
            self.renderLine(line[0], line[1], line[2])

    def clear(self):
        self.points = []
        self.lines = []

    def setXAxis(self, x):
        pass

    def setYAxis(self, y):
        pass

    def drawNN(self, nn,color=(0,0,0)):
        screen = self.screen
        w,h = screen.getSize()
        y_pad = 10
        x_pad = 10
        x_delta = (w-(2*x_pad))/len(nn.layers)
        x = x_delta/2
        layers_size = len(nn.layers)
        for i in range(layers_size):
            layer = nn.layers[i]
            y_delta = (h-(y_pad*2))/len(layer.neurons.value)
            y = y_pad+(y_delta/2)
            for j in range(len(layer.neurons.value)):
                for k in range(len(layer.weights[j])):
                    prevLayer = nn.layers[i-1]
                    temp_y_delta = (h-(y_pad*2))/len(layer.weights[j])
                    screen.line((x,y), (x-x_delta,y_pad+(temp_y_delta/2)+(temp_y_delta*k)), color)
                screen.circle((x, y), min(20, y_delta-(y_pad*2)), color)
                y+=y_delta
            x += x_delta
