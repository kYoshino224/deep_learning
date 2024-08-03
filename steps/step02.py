class Variable:
    def __init__(self, data):
        self.data = data
        
class Function:
    def __call__(self, input):
        x = input.data
        y  = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
           
import numpy as np

data = np.array([1.0, 2.0])
x = Variable(data)
#y = Function()(x)→NotImplementedError
#f=Function(), y=f(x)と同じ
f = Square()
y = f(x)
print(y.data)