import numpy as np
import sys
sys.path.append("/Users/KY/7.deeplearning/dezero_for_me")
from dezero import Variable

def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y)
z.backward()
print(x.grad, y.grad)
