#f(x, y) = (x^2 + y - 11)**2 + (x + y^2 - 7)**2

import numpy as np
import matplotlib.pyplot as plt
import torch

def f(x, y):
    return ((x**2 + y - 11)**2 + (x + y**2 - 7)**2)



x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)

X, Y = np.meshgrid(x, y)
Z = torch.tensor(f(X, Y))
"""
fig = plt.figure("Himmelblau_Function")
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
#plt.show()
"""
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([0.], requires_grad=True)

optimizer = torch.optim.Adam([x, y], lr=1e-3)

for step in range(20000):
    pre_z = f(x, y)
    optimizer.zero_grad() #loss关于weight的导数变成0.
    pre_z.backward()
    optimizer.step()

    if step % 2000 == 0:
        print("step {}: x = {}, y = {}, f(x, y) = {}".format(step, x.tolist(), y.tolist(), pre_z.item()))