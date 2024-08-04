import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creater = None

    def set_creater(self, func):
        self.creater = func
    
    def backward(self):
        f = self.creater
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward() #recuurence
        
class Function:
    def __call__(self, input):
        x = input.data
        y  = self.forward(x)
        output = Variable(y)
        output.set_creater(self) 
        self.input = input # Keep the input variable
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy



A = Square()
B= Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)


'''y.grad = np.array(1.0)
C = y.creater
b = C.input
b.grad = C.backward(y.grad)
B = b.creater
a = B.input
a.grad = B.backward(b.grad)
A = a.creater
x = A.input
x.grad = A.backward(a.grad)'''

y.grad  = np.array(1.0)
y.backward()
print(x.grad)
print(x)
print(x.grad)