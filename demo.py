from Grapher import *
from NN import *
from Data import *

data = Data()
# data.generateXORData(1000, 0.2)
data.generateCircleData(500, 0.2)
data.shuffle()

testData = Data()
# testData.generateXORData(50, 0.2)
testData.generateCircleData(50, 0.2)

nn = NN(2,4,3,1)
nn.activationFunction = "sigmoid"
nn.learningRate = 0.03

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
costGraph.setBackgroundColor((240,240,240))
screen.addSurface(costGraph, 500,0)

epoch = 0
nn.learningRate = 1
decay_steps = 20
def onUpdate():
    global epoch,decay_steps
    # nn.learningRate -= 0.007 * nn.learningRate
    nn.learningRate *= 0.96**(epoch//decay_steps)
    if(epoch//decay_steps):
        decay_steps += 20
    epoch += 1
    accuracy = nn.evaluate(testData.inputs,testData.outputs)
    print(epoch,nn.learningRate,accuracy[0])
    nn.minibatch(data,10)
    costGraph.plot(accuracy*30)
screen.onUpdate = onUpdate

screen.start()
