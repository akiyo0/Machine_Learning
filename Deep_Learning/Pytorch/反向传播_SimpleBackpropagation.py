import numpy as np

def sigmoid(x, D = False):
    if(D == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

X = np.array([[0.35], [0.9]])
y = np.array([[0.5]])

W0 = np.array([[0.1, 0.8], [0.4, 0.6]])
W1 = np.array([[0.3, 0.9]])

print('original ', W0, '\n', W1)

for j in range(100):
    #正向传播
    l0 = X
    l1 = sigmoid(np.dot(W0, l0))
    l2 = sigmoid(np.dot(W1, l1))
    l2_error = y - l2
    Error = 1 / 2.0 * (y-l2) ** 2
    print('Error:', Error)
    #######
    #back propagation

    l2_delta = l2_error * sigmoid(l2, D=True) * l1.T
    W1 += l2_delta 

    l1_error = l2_delta * W1
    l1_delta = l1_error * sigmoid(l1, D=True)
    W0 += l0.T.dot(l1_delta)
    
    #print(W0, '\n', W1)
    #######