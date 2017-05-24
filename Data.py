import random, csv
import numpy as np

#TODO: load csv file

class DataOld:
    def __init__(self):
        self.inputs = []
        self.answers = [] #TODO: change to outputs

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

    def importCSV(self, file, inputRows, outputRows, descretionize = True): #TODO: make last 2 params optional
        with open(file) as csvfile:
            r = csv.reader(csvfile)
            i = 0
            for row in r:
                if(i == 0):
                    print(row)
                    inputs = []
                    outputs = []
                    inputsName = dict()
                    outputsName = dict()
                    for j in range(len(row)):
                        if row[j] in inputRows:
                            print(row[j],"is a chosen row")
                            inputs.append(j)
                            inputsName[j] = row[j]
                        elif row[j] in outputRows:
                            outputs.append(j)
                            outputsName[j]=row[j]
                            print(row[j], "is a chosen row")
                else:
                    tinputs = []
                    tanswers =[]
                    for j in inputs:
                        tinputs.append(row[j])
                    for k in outputs:
                        tanswers.append(row[k])
                    if(len(tinputs) == len(inputRows)):
                        self.inputs.append(tinputs)
                        self.answers.append(tanswers)
                i+=1

    def normalize(self): #convert strings to number
        pass

class Data:
    def __init__(self):
        self.inputs = np.array([])
        self.outputs = np.array([])

    def generateXORData(self, samples, noise):
        padding = 0.3
        inputs = []
        outputs = []
        minimum = -10
        maximum = 10
        for i in range(samples):
            x = np.random.uniform(minimum, maximum) #TODO: profile np.random vs random
            y = np.random.uniform(minimum, maximum)
            x += padding if x > 0 else -padding
            y += padding if y > 0 else -padding
            noiseX = np.random.uniform(minimum, maximum) * noise
            noiseY = np.random.uniform(minimum, maximum) * noise
            output = 1 if x * y >= 0 else -1
            inputs.append(np.array([x,y]))
            outputs.append(np.array([output]))
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

    def generateLinerData(self, samples, noise=0.1):
        minimum = -10
        maximum = 10
        for i in range(samples):
            x = np.random.uniform(minimum, maximum) #TODO: profile np.random vs random
            y = np.random.uniform(minimum, maximum)
            x += padding if x > 0 else -padding
            y += padding if y > 0 else -padding
            noiseX = np.random.uniform(minimum, maximum) * noise
            noiseY = np.random.uniform(minimum, maximum) * noise
            output = 1 if x * y >= 0 else -1
            inputs.append(np.array([x,y]))
            outputs.append(np.array([output]))
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
