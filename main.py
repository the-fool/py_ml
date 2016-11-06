import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from n import Neuron as StepNeuron
from adalineGD import AdalineGD as GradientNeuron
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
targets = np.array([1,1,1,1,-1,-1,-1,-1])

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl)

def stepper(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

"""
For different learning rates within a range, get the average number it iterations
the neuron needs to reach perfect classification
"""
if __name__ == '__main__':
    alphas = [a for a in stepper(0.001, .03, .001)]
    iters = []
    for a in alphas:
        iter_samples = np.array([])
        # Average for a set learning rate
        for _ in range(20):
            neuron = GradientNeuron(learning_rate=a)
            neuron.learn(data, targets)
            iter_samples = np.append(iter_samples, len(neuron.error_history))
        iters.append(iter_samples.mean())

    plt.plot(alphas, iters, marker='o')
    plt.xlabel('Learning rate (alpha)')
    plt.ylabel('Iterations')
    plt.show()
