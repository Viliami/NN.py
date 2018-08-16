# NN.py
Neural Network lib in Python

**NN.py**         - Main Neural Network Class

**Grapher.py**    - Data visualization Class

**Data.py**       - Dataset Class

## Graphics
**graphics_pygame.py**   - Pygame Graphics Class

**graphics_qt.py**       - QT Graphics Class


## Install Dependencies
    pip install -r requirements.txt

## Example Code
```py
from NN import *
from Data import *

data = Data()
data.generateXORData(1000, 0.2)  #generate XOR dataset with 1000 points

testData = Data()
testData.generateXORData(50, 0.2) #generate another XOR dataset with 50 points, this will be used for validation

nn = NN(2,4,3,1) #2 input neurons, 4 neurons in first hidden layer, 3 in second hidden layer and 1 in output layer
nn.activationFunction = "sigmoid" #use the sigmoid function as the activation function
nn.learningRate = 0.03
    
for i in range(100): #100 epochs of training
    accuracy = nn.evaluate(testData)
    print(accuracy)
    nn.minibatch(data,10) #train model with minibatches of 10
```

![View](view.gif)
