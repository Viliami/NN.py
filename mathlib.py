import numpy as np
import copy
import timeit
import math

def sigmoidNew(x):
    return 1/(1+np.exp(-x))

def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoidPrime(x):
    return x*(1 - x)

class Vector:
    def __init__(self, *args):
        self.value = list(args)

    def sigmoid(self):
        return Vector(*[sigmoid(i) for i in self.value])

    def apply(self, func):
        return Vector(*[func(i) for i in self.value])

    def set(self, key, value):
        self.value[key] = value
        return value

    def __imul__(self, other):
        t = type(other)
        if(t == Vector):
            for i in range(len(self.value)):
                self.value[i] *= other.value[i]
            return self
        elif(t == int or t == float):
            for i in range(len(self.value)):
                self.value[i] *= other
            return self

    def __mul__(self, other):
        t = type(other)
        j = copy.deepcopy(self)
        if((t == list and other[0] and (type(other[0]) == int or type(other[0]) == float)) or t == tuple):
            other = Vector(*other)
            t = type(other)
        elif(t == list and other[0] and (type(other[0]) == list or type(other[0]) == tuple)):  #return an array of vectors
            m = []
            for i in range(len(other)):
                m.append(self * other[i])
            return Vector(*m)

        if(t == Vector):
            for i in range(len(j.value)):
                j.value[i] *= other.value[i]
            return j
        elif(t == int or t == float):
            for i in range(len(j.value)):
                j.value[i] *= other
            return j
        elif(t == Matrix):
            return other * self

    def __truediv__(self, other):
        t = type(other)
        if(t == int or t == float):
            j = copy.deepcopy(self)
            for i in range(len(j.value)):
                j.value[i] /= other
            return j

    def __iadd__(self, other):
        t = type(other)
        if(t == Vector):
            for i in range(len(self.value)):
                self.value[i] += other.value[i]
            return self
        elif(t == int):
            for i in range(len(self.value)):
                self.value[i] += other
            return self

    def __add__(self, other):
        t = type(other)
        j = copy.deepcopy(self)
        if(t == Vector):
            for i in range(len(j.value)):
                j.value[i] += other.value[i]
            return j
        elif(t == int):
            for i in range(len(j.value)):
                j.value[i] += other
            return j

    def __isub__(self, other):
        t = type(other)
        if(t == Vector):
            for i in range(len(self.value)):
                self.value[i] -= other.value[i]
            return self
        elif(t == int):
            for i in range(len(self.value)):
                self.value[i] -= other
            return self

    def __sub__(self, other):
        t = type(other)
        j = copy.deepcopy(self)
        if(t == Vector):
            for i in range(len(j.value)):
                j.value[i] -= other.value[i]
            return j
        elif(t == int):
            for i in range(len(j.value)):
                j.value[i] -= other
            return j

    def __str__(self):
        return str(self.value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        return self.value[key]

class Matrix:
    def __init__(self, listOrxSize, ySize=None):
        t = type(listOrxSize)
        self.value = []
        if(t == list):
            t = type(listOrxSize[0])
            if(t == list or t == tuple):
                for i in range(len(listOrxSize)): #TODO: fix this up
                    self.value.append(Vector(*listOrxSize[i]))
            elif(t == Vector):
                self.value = listOrxSize
        elif(t == int):
            self.value = [Vector(*[0]*listOrxSize) for i in range(ySize)]

    def transpose(self):
        return Matrix(list(zip(*[x.value for x in self.value])))

    def sum(self):
        total = 0
        for v in self.value:
            total += sum(v)
        return total

    def apply(self, func):
        m = copy.deepcopy(self)
        for v in m.value:
            for i in range(len(v)):
                v.set(i, func(v[i]))
        return m

    def set(self, j, value):
        t = type(value)
        if(t == Vector):
            self.value[j] = value
        elif(t == int):
            self.value[j] = Vector(value)
        elif(t == list or t == tuple):
            self.value[j] = Vector(*value)

    def __len__(self):
        return len(self.value)

    def __str__(self):
        for i in range(len(self.value)):
            print(self.value[i])
        return ""

    def __add__(self, other):
        t = type(other)
        if(t == Matrix):
            m = copy.deepcopy(self)
            for i in range(len(self.value)):
                m.set(i, self[i]+other[i])
            return m
        elif(t == Vector):
            pass #TODO
        elif(t == int):
            m = copy.deepcopy(self)
            for i in range(len(m.value)):
                m.set(i, self[i]+other)
            return m
        elif(t == tuple or t == list):
            pass #TODO

    def __iadd__(self, other):
        t = type(other)
        if(t == Matrix):
            for i in range(len(self.value)):
                self.set(i, self[i]+other[i])
            return self
        elif(t == Vector):
            pass #TODO
        elif(t == int):
            pass #TODO
        elif(t == tuple or t == list):
            pass #TODO

    def __sub__(self, other):
        t = type(other)
        if(t == Vector):
            m = copy.deepcopy(self)
            for i in range(len(m.value)):
                m.set(i, m[i]-other[i])
            return m

    def __isub__(self, other):
        pass

    def __mul__(self, other):
        t = type(other)
        if(t == Vector):
            temp = []
            for i in self:
                temp.append(i.value)
            return Vector(*np.dot(temp,other)) #TODO, fix this (not always vector)
        elif(t == int):
            m = copy.deepcopy(self)
            for i in m.value:
                for j in range(len(i)):
                    i.set(j, i[j] * other)
            return m
        elif(t == Matrix):
            m = copy.deepcopy(self)
            for i in range(len(m.value)):
                m.set(i, m[i]*other[i])
            return m

    def __getitem__(self, key):
        return self.value[key]

class Matrix2:
    def __init__(self,dimensions):
        self.dimensions = dimensions
        self.value = np.random.rand(*dimensions)

    def randomize(self, mu, sigma):
        self.value = np.random.normal(mu, sigma, self.dimensions)
        return self

    def __mul__(self,other):
        t = type(other)
        if(t is np.ndarray):
            return self.value*other
        elif(t is Matrix2 or t is Vector2):
            return self.value*other.value

    def sum(self):
        return self.value.sum()

    def apply(self, func):
        grid = np.meshgrid(self.value)
        self.value = np.vectorize(func)(*grid)
        return self

    def transpose(self):
        return self.value.T

    def __getitem__(self, key):
        return self.value[key]

    def __len__(self):
        return self.dimensions[0]

class Vector2:
    def __init__(self,*args):
        self.value = np.array(args,dtype=np.float64)

    def __mul__(self,other):
        t = type(other)
        if(t is Vector2):
            return self.value*other.value
        elif(t is np.ndarray):
            return self.value*other

    def sum(self):
        return self.value.sum()

    def set(self, key, value):
        self.value[key] = value
        return self

    def __getitem__(self, key):
        return self.value[key]

    def __iadd__(self, other):
        t = type(other)
        if(t is Vector2):
            return Vector2(*(self.value+other.value))

    def __add__(self, other):
        t = type(other)
        if(t is Vector2):
            return Vector2(*(self.value+other.value))

    def __sub__(self, other):
        t = type(other)
        if(t is Vector2):
            return Vector2(*(self.value-other.value))

    def apply(self,func):
        for i in range(len(self.value)):
            self.value[i] = func(self.value[i])
        return self

    def __len__(self):
        return len(self.value)

    def __str__(self):
        return str(self.value)
