import pygame, pygame.gfxdraw

BLACK = (0,0,0)
WHITE = (255,255,255)

def frange(start,end,step):
    return map(lambda x: x*step, range(int(start*1./step),int(end*1./step)))

class BaseGraph:
    def __init__(self, screen, backgroundColor=WHITE):
        self.screen = screen
        self.backgroundColor = backgroundColor

    def changeWidth(self,width):
        # self.screen.surface = pygame.transform.scale(self.screen,(width,self.height))
        pass

    def setHeight(self,height):
        # self.screen = pygame.transform.scale(self.screen,(self.width,height))
        pass

    def clear(self):
        self.screen.clear(self.backgroundColor)

    def render(self):
        pass

class Grapher(BaseGraph):
    def __init__(self, screen, gridWidth=10, gridHeight=10, gridShown=False, radius=4):
        self.screen = screen
        self.width, self.height =  screen.getSize()
        self.radius = radius
        self.gridShown = gridShown
        self.gridWidth,self.gridHeight = gridWidth,gridHeight
        self.points = []
        self.lines = []
        self.backgroundColor = WHITE
        self.gridColor = BLACK
        self.scaleX, self.scaleY = (1,1)

    def showGrid(self):
        self.gridShown = True
    def hideGrid(self):
        self.gridShown = False

    def setGridColor(self, color):
        self.gridColor = color

    def setBackgroundColor(self, color):
        self.backgroundColor = color

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
            # pygame.gfxdraw.aacircle(self.surface, int(sPos[0]), int(sPos[1]), int(radius), color)
            # pygame.gfxdraw.filled_circle(self.surface, int(sPos[0]), int(sPos[1]), int(radius), color)
            self.screen.circle(sPos,radius,color)

    def renderPoint(self, x, y, color,circle=True):
        if(self.radius > 1):
            # pygame.gfxdraw.aacircle(self.surface, int(x),int(y),self.radius, (color[0],color[1],color[2]))
            # pygame.gfxdraw.filled_circle(self.surface, int(x),int(y),self.radius, color)
            x,y = self.graphToSurface((x,y))
            if(circle):
                self.screen.circle((x,y),self.radius,color,True)
            else:
                self.screen.rectangle(x,y,self.radius,self.radius,color)
    def renderLine(self, startPos, endPos, color):
        # pygame.draw.aaline(self.surface, color, startPos, endPos)
        sPos = self.graphToSurface(startPos)
        ePos = self.graphToSurface(endPos)
        self.screen.line(sPos,ePos,color)

    def renderGrid(self):
        gWidth=float(self.width)/self.gridWidth
        gHeight=float(self.height)/self.gridHeight
        for x in frange(1,self.gridWidth,self.scaleX):
            # pygame.draw.line(self.screen, BLACK, (x*gWidth, 0),(x*gWidth, self.height))
            self.screen.line((x*gWidth,0),(x*gWidth,self.height), self.gridColor)

        for y in range(self.gridHeight):
            # pygame.draw.line(self.surface, BLACK, (0, y*gHeight),(self.width, y*gHeight))
            self.screen.line((0,y*gHeight),(self.width,y*gHeight),self.gridColor)

    def render(self, connected=False,text=None): #TODO: connect the lines
        if(self.gridShown):
            self.renderGrid()

        for line in self.lines:
            self.renderLine(line[0], line[1], line[2])
        if(not connected):
            for point in self.points:
                self.renderPoint(point[0], point[1], point[2])
        else:
            # for i in range(1,len(self.points)):
            #     point = self.points[i]
            #     prevPoint = self.points[i-1]
                # self.renderLine((point[0],point[1]),(prevPoint[0],prevPoint[1]),BLACK)
            if(len(self.points) > 1): #TODO: optimize this
                pygame.draw.aalines(self.screen.screen,BLACK, False,[self.graphToSurface((point[0],point[1])) for point in self.points])

            if(text):
                self.screen.text(text,(0,0))

    def clearPoints(self):
        self.points = []
        self.lines = []

    def setXAxis(self, x):
        self.gridWidth = x

    def setYAxis(self, y):
        self.gridHeight = y

    def setAxis(self, x,y):
        self.gridWidth,self.gridHeight = x,y

    def setScaleX(self, scale):
        self.scaleX = scale

    def setScaleY(self, scale):
        self.scaleY = scale

    def plotSeries(self, y):
        self.plot(self.gridWidth,y,(255,0,0))
        self.gridWidth += 1
        if(self.gridWidth%50 == 0):
            self.setScaleX(self.gridWidth/10)

class Structure(BaseGraph):
    def render(self, nn,color=(66, 235, 244)): #TODO: add value on hover
        screen = self.screen
        screen.clear(self.backgroundColor)
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
                ncolor = color
                if(layer.neurons[j] < 0):
                    ncolor = (200,0,0)
                screen.circle((x, y), min(20, y_delta-(y_pad*2)), ncolor,True)
                for k in range(len(layer.weights[j])):
                    prevLayer = nn.layers[i-1]
                    temp_y_delta = (h-(y_pad*2))/len(layer.weights[j])
                    weight = layer.weights[j][k]
                    wcolor = BLACK
                    if(weight < 0):
                        wcolor = (200,0,0)
                    screen.line((x,y), (x-x_delta,y_pad+(temp_y_delta/2)+(temp_y_delta*k)), wcolor, weight)

                y+=y_delta
            x += x_delta
        screen.text("Network structure",(0,0))

class NeuralGrid(Grapher): #only possible if there are 2 input neurons and 1 output
        def render(self, nn):
            if(self.gridShown):
                self.renderGrid()
            for x in range(self.gridWidth):
                for y in range(self.gridHeight+1):
                    output = nn.predict([x,y])[0]
                    c = max(0,output*255)
                    c = min(255,c)
                    self.renderPoint(x,y,(0,c,c),False)
