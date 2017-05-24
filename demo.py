from Grapher import *
from NN import *
from Data import *

data = Data()
data.generateXORData(100, 0.2)

nn = NN(2,4,3,1)
nn.activationFunction = "tanh"
nn.learningRate = 0.0009

SCREEN_WIDTH, SCREEN_HEIGHT = (900,600)
screen = Screen(SCREEN_WIDTH,SCREEN_HEIGHT,"Test")

#draw neural network structure
structure = Structure(500, 300, nn)
screen.addSurface(structure, 0, 0)

#add 2d neural grid vizualisation
axis = 10
ngrid = NeuralGrid(200, 200, np.linspace(-axis,axis,axis*2), np.linspace(-axis,axis,axis*2), nn, data)
screen.addSurface(ngrid, 50, 300)

#add time series visualisaiton
costGraph = TimeSeries(200,200, np.linspace(0,10,10))
screen.addSurface(costGraph, 500,0)

epoch = 0
def onUpdate():
    global epoch
    epoch += 1
    print(epoch)
    # nn.train([[1,1],[2,2],[3,3],[3,4]],[[1],[1],[1],[0]])
    # nn.train(data)
    # costGraph.plot(nn.evaluate(np.array([[1,1],[2,2],[3,3],[3,4]]),np.array([[1],[1],[1],[0]]))*35)
    nn.train(data)
    costGraph.plot(nn.evaluate(data.inputs,data.outputs)*30)
    # print(nn.evaluate(data.inputs,data.outputs))
screen.onUpdate = onUpdate

screen.start()
