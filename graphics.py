import pygame, os, pygame.gfxdraw
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (20,50)
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
pygame.init()

#TODO: make a screen draggable option

class Screen:
    def __init__(self,width, height):
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.screen = pygame.Surface((width,height))
        self.fontColor = BLACK #default font color
        self.font = pygame.font.SysFont("Arial",16)

    def setDisplay(self,caption="Graphics Window"):
        self.screen = pygame.display.set_mode((self.width,self.height))
        pygame.display.set_caption(caption)

    def hEvents(self):
        for e in pygame.event.get():
            if(e.type == pygame.QUIT):
                return False
            elif(e.type == pygame.KEYDOWN):
                if(e.key == pygame.K_ESCAPE):
                    return False
        return True

    def clear(self,color=WHITE):
        self.screen.fill(color)

    def update(self):
        pygame.display.flip()

    def quit(self):
        pygame.quit()

    def getSize(self):
        return (self.width,self.height)

    def circle(self,pos, rad, color):
        x,y = pos
        x = int(x)
        y = int(y)
        rad = int(rad)
        pygame.gfxdraw.filled_circle(self.screen, x,y,rad, (66, 235, 244))
        pygame.gfxdraw.aacircle(self.screen, x,y,rad, BLACK)

    def line(self,sPos, ePos, color):
        pygame.draw.aaline(self.screen, color, sPos, ePos)

    def blit(self, screen, dest=(0,0),area=None):
        self.screen.blit(screen.screen, dest, area=None)

    def text(self,text, pos, color=BLACK, aa=True):
        self.screen.blit(self.font.render(text, aa, color), pos)
