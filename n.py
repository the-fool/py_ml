import numpy as np

class Neuron:
    def __init__(self, learning_rate=0.1, max_iters=1000):
        self.learning_rate = learning_rate
        self.max_iters = max_iters

    def learn(self, inputs, targets):
        """
        input: matrix of [samples, features]
        targets: vector of [samples]
        """
        n_features = inputs.shape[1]
        self.weights = np.random.uniform(-1/n_features, 1/n_features, 1 + n_features)
        self.errors = []

        for _ in range(self.max_iters):
            errors = 0
            for sample, target in zip(inputs, targets):
                delta = self.learning_rate * (target - self.predict(sample))
                self.weights[1:] += delta * sample
                self.weights[0] += delta
                errors += int(delta != 0.0)
            self.errors.append(errors)
            if errors == 0.0:
                break
        return self

    def predict(self, sample):
        return np.where(self.net_input(sample) >= 0.0, 1, -1)

    def net_input(self, sample):
        return np.dot(sample, self.weights[1:]) + self.weights[0]
