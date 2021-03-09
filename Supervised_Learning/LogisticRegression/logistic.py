import numpy as np
def sigmoid(x):
    return 1/(1 + np.exp(-x))

print(sigmoid(1000))

def 