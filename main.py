import matplotlib.pyplot as plt
import numpy as np
from n import Neuron

data = np.array([
    # Black
    [0.1, 1],
    [0.33, 0.98],
    [0.7, 0.67],
    [0.9, 0.4],
    # White
    [0.05, 0.72],
    [0.25, 0.55],
    [0.42, 0.12],
    [0.6, 0.3]
])
targets = np.array([-1,-1,-1,-1,1,1,1,1])
def stepper(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

"""
For different learning rates within a range, get the average number it iterations
the neuron needs to reach perfect classification
"""
alphas = [a for a in stepper(0.01, 4, .04)]
iters = []
for a in alphas:
    iter_samples = np.array([])
    # Average for a set learning rate
    for _ in range(20):
        neuron = Neuron(learning_rate=a)
        neuron.learn(data, targets)
        iter_samples = np.append(iter_samples, neuron.errors.index(0.0))
    iters.append(iter_samples.mean())

plt.plot(alphas, iters, marker='o')
plt.xlabel('Learning rate (alpha)')
plt.ylabel('Iterations')
plt.show()
