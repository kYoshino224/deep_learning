import numpy as np
import unittest
import weakref
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
        self.generation = 0

    def set_creater(self, func):
        self.creater = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad = False):
        if self.grad == None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()

        def add_func(func):
            if func not in seen_set:
                funcs.append(func)
                seen_set.add(func)
                funcs.sort(key=lambda x: x.generation)    

        add_func(self.creater)
        while funcs:
            func = funcs.pop()
            gys = [output().grad for output in func.outputs]
            gxs = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creater is not None:
                    add_func(x.creater)
            
            if not retain_grad:
                for y in func.outputs:
                    y().grad = None

    def clear_grad(self):
        self.grad = None
        
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys  = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creater(self)
        self.inputs = inputs # Keep the input variable
        self.outputs = outputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    def backward(self, gy):
        x = self.inputs[0].data
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
x0 = Variable(np.array([1.0]))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)