class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y  = x ** 2
        output = Variable(y)
        return output
           
import numpy as np

data = np.array([1.0, 2.0])
x = Variable(data)
#f=Function(), y=f(x)と同じ
y = Function()(x)
print(y.data)