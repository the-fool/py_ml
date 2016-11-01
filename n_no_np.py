import random

class Neuron:

    def __init__(self, learning_rate=0.01, max_iters=300):
        self.learning_rate = learning_rate
        self.max_iters = max_iters

    def learn(self, test_set, target_vector):
        # initialize weights
        n_features = len(test_set[1])
        self.weights = [random.uniform((-2.4)/n_features, 2.4/n_features) for _ in range(n_features)]
        self.threshold = random.uniform(-1, 1)
        self.error_history = []
        self.num_error_history = []

        for _ in range(self.max_iters):
            total_error = 0
            num_errors = 0
            for xi, target in zip(test_set, target_vector):
                adjust = self.learning_rate * (target - self.compute(xi))
                self.weights = [w + adjust * x for w, x in zip(self.weights, xi)]
                self.threshold += adjust * self.learning_rate * self.learning_rate
                total_error += (adjust ** 2)
                num_errors += int(adjust != 0.0)
                print(self.weights)
            self.num_error_history.append(num_errors)
            self.error_history.append(total_error)

    def compute(self, x):
        return 1 if self.dot(x) > self.threshold else 0

    def dot(self, x):
        return sum([i * j for i,j in zip(x, self.weights)])



if __name__ == '__main__':
    data = [
    [1,2,3,4]
    [2,3,4,5]
    ]
