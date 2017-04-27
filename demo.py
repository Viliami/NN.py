from NN import *
from graphics import *
import Grapher as graph
import random
import Data

#TODO:add momentum
#TODO: add dropout

def linear(x,y):
    return (x >= y)

def randomData(x,y):
    return random.uniform(0,1)

data = []
answers = []
target = [0.3]

nn = NN(2, [3,3], 1)

nn.setActivation("tanh")
nn.setLearningRate(0.01)

screen = Screen(700,600)
screen.setDisplay("NN.py")
nnStructure = graph.Structure(Screen(300,300))
nnStructure2 = graph.Grapher(Screen(300,150),5,2,False)
gridX, gridY = 20,20

grid = graph.NeuralGrid(Screen(300,300),gridX,gridY,False,300.0/gridX - 0)
dataGrid = graph.Grapher(Screen(300,300),gridX,gridY,True,5)

data = Data.Data()
data.populate(linear, 40,[(0,gridX),(0,gridY)])

for i in range(len(data.inputs)):
    dataGrid.plot(data.inputs[i][0],data.inputs[i][1], (data.answers[i][0]*255,0,0))

while(screen.hEvents()):
    # screen.clock.tick(10)
    screen.clear()
    nnStructure.clear()
    nnStructure2.clear()
    grid.clear()
    dataGrid.clear()

    # nn.train([[0.2,0.2]],[target])
    # nn.train(data,answers)
    nn.train(data.inputs,data.answers,2)
    nnStructure2.plotSeries(nn.cost(data.answers[-1])+1)

    nnStructure.render(nn)
    nnStructure2.render(True)
    grid.render(nn)
    dataGrid.render(False)

    screen.blit(nnStructure.screen,(20,0))
    screen.blit(nnStructure2.screen,(320,20))
    screen.blit(grid.screen,(330,295))
    screen.blit(dataGrid.screen,(20,300))
    screen.update()

screen.quit()
