import numpy as np
import sys
sys.path.append("/Users/KY/7.deeplearning/dezero_for_me")
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()
print(y)
print(x.grad)