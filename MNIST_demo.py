from Grapher2 import *
from NN3 import *
from Data import *
from scipy import ndimage, signal

screen = Screen(300,200,"Graph")

nn = NN(784,40,10)
nn.activationFunction = "sigmoid"
nn.learningRate = 1

# trainingData = Data()
# trainingData.importMNIST("/home/viliami/Documents/Programming/Python/NN.py/MNIST/","training")
# trainingData.shuffle()

testData = Data()
testData.importMNIST("/home/viliami/Documents/Programming/Python/NN.py/MNIST/","test")

costGraph = TimeSeries(200,200, np.linspace(0,10,10))
costGraph.setBackgroundColor((240,240,240))
screen.addSurface(costGraph, 0,0)

def convolve(a,b):
    # return np.convolve(a,b,"valid")
    #  return np.fft.irfft2(np.fft.rfft2(a)*np.fft.rfft2(b,a.shape))
    #  return ndimage.convolve(a,b)
     return signal.fftconvolve(a,b,mode="valid")

def maxpool(a, downsample=(2,2)):
    h, w = a.shape
    return a.reshape(h // downsample[1], downsample[1], w // downsample[0], downsample[0]).max(axis=(1,3))

flter = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
swag = convolve(testData.inputs[0].reshape(28,28),flter)
swag2 = maxpool(swag)
print("new shape",swag2.shape)

# img = ImgArray(100,200,testData.inputs[0])
img = ImgArray(100,200,swag2)
img.toRGB()
img.backgroundColor = (100,100,100)
screen.addSurface(img, 200,0)

epoch = 0
err = nn.test(testData)
costGraph.plot(err[0]/err[1]*10)

nn.test(testData)

# for i in range(50):
def onUpdate():
    return
    global epoch
    epoch += 1
    if(epoch > 40):
        nn.learningRate = 0.3
    nn.minibatch(trainingData,1000)
    # print(epoch, nn.evaluate(data.inputs, data.outputs))
    err = nn.test(testData)
    print(epoch, err)
    costGraph.plot(err[0]/err[1]*10)
screen.onUpdate = onUpdate

screen.start()
