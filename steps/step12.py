import numpy as np
import unittest

def as_array(t):
            if np.isscalar(t):
                print('Warning: Use ndarray instead of scalar!')
                return np.array(t)
            return t

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        self.data = data
        self.grad = None
        self.creater = None

    def set_creater(self, func):
        self.creater = func
    
    def backward(self):
        if self.grad == None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creater]
        while funcs:
            func = funcs.pop()
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)
            if x.creater is not None:
                funcs.append(x.creater)
        
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys  = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creater(self)
        self.inputs = inputs # Keep the input variable
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
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
def square(x):
    return Square()(x)  
  
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy
def exp(x):
    return Exp()(x)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y, )
    def backward(self, gy):
        return gy, gy
def add(x0, x1):
    return Add()(x0, x1)

# Testing
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(4.0))
y = add(x0, x1)
print('y:', y.data)

z = square(y)
print('z:', z.data)