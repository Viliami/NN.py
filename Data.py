import random, csv, struct
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

    def shuffle(self):
        batch = np.random.permutation(len(self.outputs))
        self.outputs = self.outputs[batch]
        self.inputs = self.inputs[batch]

    def importMNIST(self, path, trainingOrTest="test"):
        if(trainingOrTest == "training"):
            images = "train-images.idx3-ubyte"
            labels = "train-labels.idx1-ubyte"
        else:
            images = "t10k-images.idx3-ubyte"
            labels = "t10k-labels.idx1-ubyte"

        with open(path+images, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            self.inputs =  np.fromstring(f.read(), dtype=np.uint8).reshape(shape[0],shape[1]*shape[2])

        with open(path+labels, 'rb') as fg:
            magic, num = struct.unpack(">II", fg.read(8))
            self.outputs = np.fromfile(fg, dtype=np.uint8)

        self.outputs = np.array([self.vectorizeResult(i) for i in self.outputs]).reshape(len(self.outputs),10)

    def vectorizeResult(self, i):
        e = np.zeros((10,1))
        e[i] = 1.0
        return e


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
            output = 1 if x * y >= 0 else 0
            inputs.append(np.array([x,y]))
            outputs.append(np.array([output]))
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

    def generateCircleData(self, samples, noise=0.1):
        minimum = -10
        maximum = 10
        radius = 10
        inputs = []
        outputs = []
        for i in range(samples):
            r = np.random.uniform(0, radius)
            angle = np.random.uniform(0, 2*np.pi)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            noiseX = np.random.uniform(minimum, maximum) * noise
            noiseY = np.random.uniform(minimum, maximum) * noise
            # output = 1 if x * y >= 0 else -1
            output = int(self.distance((x+noiseX,y+noiseY),(0,0)) < (radius * 0.5))
            inputs.append(np.array([x+noiseX,y+noiseY]))
            outputs.append(np.array([output]))
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

    def distance(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
