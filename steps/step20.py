import numpy as np
import unittest
import weakref
import contextlib

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

class Config:
    enable_backprop = True

def as_array(t):
            if np.isscalar(t):
                print('Warning: Use ndarray instead of scalar!')
                return np.array(t)
            return t

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        self.data = data
        self.grad = None
        self.creater = None
        self.generation = 0
        self.name = name

    @property
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' *9)
        return 'variable(' + p + ')'


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

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creater(self)
            self.inputs = inputs # Keep the input variable
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
Variable.__add__ = add

class Mull(Function):
    def forward(self, x0, x1):
        return x0 * x1
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
def mul(x0, x1):
    return Mull()(x0, x1)
Variable.__mul__ = mul

# Testing
a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = a * b + c
y.backward()

print(y.data)
print(a.grad)
print(b.grad)
print(c.grad)