import pygame, os, pygame.gfxdraw,math
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (20,50)
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

class Screen:
    def __init__(self, width, height, title="Untitled"):
        self.clock = pygame.time.Clock()
        self.width, self.height = width, height
        self.surfaces = []
        pygame.init()
        self.font = pygame.font.SysFont("Arial",16)
        self.screen =pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.fps = 60

    def setFPS(self, fps):
        self.fps = fps

    def onUpdate(self):
        pass

    def hEvents(self):  
        self.onUpdate()
        for e in pygame.event.get():
            if(e.type == pygame.QUIT):
                return False
            elif(e.type == pygame.KEYDOWN):
                if(e.key == pygame.K_ESCAPE):
                    return False
        return True

    def getSize(self):
        return (self.width,self.height)

    def start(self):
        while self.hEvents():
            self.clock.tick(self.fps)
            self.screen.fill(WHITE)
            for surface in self.surfaces:
                # surface.fill(WHITE)
                surface.render()
                self.screen.blit(surface,(surface.x,surface.y))
            pygame.display.flip()
        pygame.quit()

    def addSurface(self, surface, x, y):
        surface.x, surface.y = x,y
        self.surfaces.append(surface)

class Surface(pygame.Surface): #TODO: profile blitting vs direct drawing
    def __init__(self, width, height):
        super().__init__((width+1,height+1))
        self.width, self.height = width, height
        self.x, self.y = 0,0
        self.parent = None
        self.font = pygame.font.SysFont("Arial",16)
        self.backgroundColor = WHITE

    def render(self):
        self.fill(self.backgroundColor)
        self.filledSquare((self.x+20,self.y+20),50)
        self.filledRectangle((self.x,self.y),(50,50),WHITE)
        self.ellipse((10,10),(15,10))
        self.circle((30,30),15,(255,0,0))
        self.line((10,10),(self.x, self.y),BLACK,2)
        self.text((100,200),"ayylmao")

    def rectangle(self, pos, size, color=BLACK):
        pygame.draw.rect(self, color, (*pos,*size),1)

    def filledRectangle(self, pos, size, color=BLACK, outlineColor=BLACK):
        pygame.draw.rect(self, outlineColor, (*pos,*size),1)
        self.fill(color, (pos[0]+1, pos[1]+1, size[0]-2, size[1]-2))

    def square(self, pos, width, color=BLACK):
        pygame.draw.rect(self, color, (*pos, width, width), 1)

    def filledSquare(self, pos, width, color=BLACK, outlineColor=BLACK):
        pygame.draw.rect(self, outlineColor, (*pos, width, width),1)
        self.fill(color, (pos[0]+1, pos[1]+1, width-2, width-2))

    def ellipse(self, pos, size, color=BLACK):
        pygame.draw.ellipse(self, color, (pos[0]-size[0], pos[1]-size[1], size[0]*2, size[1]*2), 1)

    def filledEllipse(self, pos, size, color=BLACK, outlineColor=BLACK):
        w,h = size[0]*2, size[1]*2
        pygame.draw.ellipse(self, color, (pos[0]+1, pos[1]+1, w-2, h-2))
        pygame.draw.ellipse(self, outlineColor, (*pos, w, h), 1)

    def circle(self, pos, rad, color=BLACK):
        pygame.draw.circle(self, color, pos, rad, 1)

    def filledCircle(self, pos, rad, color=BLACK, outlineColor=BLACK):
        pygame.draw.circle(self, color, pos, rad-1)
        pygame.draw.circle(self, outlineColor, pos, rad, 1)

    def line(self, sPos, ePos, color=BLACK, width=1):
        pygame.draw.aaline(self, color, sPos, ePos, width)

    def text(self, pos, text, color=BLACK):
        self.blit(self.font.render(text, True, color), pos)

    def imageArray(self, array):
        pygame.surfarray.blit_array(self, array)
