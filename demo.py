from NN import *
import graphics as g

#TODO:add momentum

target = [3,6]
samples = 100
data = [[0.2,0.2]]*samples
answers = [target]*samples

nn = NN(2, [2], 2)
nn.setLearningRate(0.5)
nn.train(data, answers,1,True)
print(nn.predict([0.2,0.2]))

screen = g.init(700,350, "NN.py")
while(g.hEvents()):
    g.begin()

    nn.draw(screen)

    g.end()
g.quit()
