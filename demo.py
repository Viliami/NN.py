from NN import *
import graphics as g

target = [0,1,2]
nn = NN(3, [2,2], 3)
nn.feedForward()

samples = 500
for i in range(samples):
    nn.backprop(target)
    nn.feedForward()
    if(not (i+1)%(samples/10)):
        print(i+1,"tries, error:",nn.cost(target))

screen = g.init(700,350, "NN.py")
while(g.hEvents()):
    g.begin()

    nn.draw(screen)

    g.end()
g.quit()
