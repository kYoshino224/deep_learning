import numpy as np

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
    def __call__(self, input):
        x = input.data
        y  = self.forward(x)
        def as_array(x):
            if np.isscalar(x):
                print('Warning: Use ndarray instead of scalar!')
                return np.array(x)
            return x
        output = Variable(as_array(x))
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



x = Variable(np.array(0.5))
'''a = square(x)
b = exp(a)
y = square(b)'''
y = square(exp(square(x)))

y.grad  = np.array(1.0)
y.backward()
print(x.grad)