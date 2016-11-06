import numpy as np

class AdalineGD:
    def __init__(self, learning_rate=0.01, max_iters=500):
        self.learning_rate = learning_rate
        self.max_iters = max_iters

    def learn(self, inputs, targets):
        n_features = inputs.shape[1]
        self.weights = np.random.uniform(-2.4/n_features, 2.4/n_features, 1 + n_features)
        self.error_history = []

        for i in range(self.max_iters):
            output = self.net_input(inputs)
            errors = targets - output
            self.weights[1:] += self.learning_rate * inputs.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            squared_error = (errors ** 2).sum()
            self.error_history.append(squared_error)
            print(squared_error)
            if squared_error < 0.01:
                break
        return self

    def net_input(self, inputs):
        return np.dot(inputs, self.weights[1:]) + self.weights[0]

    def activation(self, inputs):
        return self.net_input(inputs)

    def predict(self, inputs):
        return np.where(self.activation(inputs) >= 0.0, 1, -1)
