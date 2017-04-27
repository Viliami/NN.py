import random

class Data:
    def __init__(self):s
        self.inputs = []
        self.answers = []

    #generate n sample data at random points based on a custom function
    def populate(self, func, samples,inputRange):
        gridX = 10
        gridY = 10
        for i in range(samples):
            inputs = []
            for x in inputRange:
                inputs.append(random.uniform(x[0],x[1]))

            self.inputs.append(inputs)
            self.answers.append([func(*inputs)])

    def importCSV(self):
        pass
