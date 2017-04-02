import pygame, os, pygame.gfxdraw

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (10,10)
screen = 0
clock = 0
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

def init(width, height, caption="Graphics Window"):
    global screen
    pygame.init()
    pygame.display.set_caption(caption)
    # clock = pygame.time.Clock()
    screen = pygame.display.set_mode((width, height))
    return screen

def hEvents():
    for e in pygame.event.get():
        if(e.type == pygame.QUIT):
            return False
        elif(e.type == pygame.KEYDOWN):
            if(e.key == pygame.K_ESCAPE):
                return False
    return True

def begin(color=WHITE):
    screen.fill(color)

def end():
    pygame.display.flip()

def quit():
    pygame.quit()

def circle((x,y), rad, color):
    pygame.gfxdraw.aacircle(screen, x, y, rad, color)
    pygame.gfxdraw.filled_circle(screen, x,y,rad, color)
