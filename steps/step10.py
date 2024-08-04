import numpy as np
import unittest

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
        def as_array(t):
            if np.isscalar(t):
                print('Warning: Use ndarray instead of scalar!')
                return np.array(t)
            return t
        output = Variable(as_array(y))
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

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data + eps)
    x1 = Variable(x.data  - eps)
    return (f(x0).data - f(x1).data) / (2 * eps)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(4.0))
        y = square(x)
        expected = np.array(16.0)
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

unittest.main()
