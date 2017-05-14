from Grapher import *
from NN import *

SCREEN_WIDTH, SCREEN_HEIGHT = (900,600)
screen = Screen(SCREEN_WIDTH,SCREEN_HEIGHT,"Test")
nn = NN(2,3,1)

#draw neural network structure
structure = Structure(500, 300, nn)
screen.addSurface(structure, 0, 0)

#add 2d neural grid vizualisation
ngrid = NeuralGrid(200, 200, np.linspace(0,10,10), np.linspace(0,10,10), nn)
screen.addSurface(ngrid, 50, 300)

def onUpdate():
    nn.train([[1,1],[2,2],[3,3],[3,4]],[[1],[1],[1],[0]])
screen.onUpdate = onUpdate

screen.start()
