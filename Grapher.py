import pygame, pygame.gfxdraw

BLACK = (0,0,0)
WHITE = (255,255,255)

class Grapher:
    def __init__(self, surface, gridWidth=10, gridHeight=10, gridShown=False, radius=4):
        self.surface = surface
        self.width, self.height =  surface.get_size()
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

    def linePoint(self, m, c, x): #gets y at a point x
        return m*x + c

    def surfaceToGraph(self, (x,y)):
        y = (self.height-y)
        gWidth = float(self.width)/self.gridWidth
        gHeight = float(self.height)/self.gridHeight
        return (x/gWidth,y/gHeight)

    def graphToSurface(self, (x,y)):
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
        for x in xrange(1,self.gridWidth):
            pygame.draw.line(self.surface, BLACK, (x*gWidth, 0),(x*gWidth, self.height))
        for y in xrange(1, self.gridHeight):
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
