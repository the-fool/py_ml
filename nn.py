import random
import math

# -- Utility functions -- #
def sigmoid(i):
    return 1 / (1 + math.exp(-i))

def dot(v, w):
    return sum([i*j for i,j in zip(v,w)])


class Neuron:
    def __init__(self, n_inputs = 1, learning_rate = 0.3):
        bound = 2.4/n_inputs
        self.weights = [random.uniform(-bound, bound) for _ in range(n_inputs)]
        self.previous_delta = 0
        self.threshold = random.uniform(-bound, bound)
        self.learning_rate = learning_rate
        self.adaptive_factor = 1

    def adjust_weight_and_threshold(self, delta, inputs_vector):
        momentum = 0.5 * self.previous_delta
        self.adaptive_factor += (delta - self.previous_delta)
        self.threshold -= self.learning_rate * delta
        for i, input_value in enumerate(inputs_vector):
            # Add adjustment plus a momentumm factor
            self.weights[i] += (self.learning_rate * delta * input_value * self.adaptive_factor + momentum)
        self.previous_delta = delta

    def output(self, input_vector):
        return sigmoid(dot(input_vector, self.weights) - self.threshold)

# -- Neural network functions -- #

def feed_forward(n_network, input_v):
    layered_outputs = []
    for layer in n_network:
        output_v = [neuron.output(input_v) for neuron in layer]
        layered_outputs.append(output_v)
        input_v = output_v
    return layered_outputs

def back_propagate(neural_network, input_v, target):
    outer_layer = neural_network[1]
    hidden_layer = neural_network[0]
    hidden_output_v, final_output_v = feed_forward(neural_network, input_v)

    # get adjustment vector for output layer
    error_v = [target - output for output in final_output_v]
    error_adjusts = [error * output * (1 - output) for error, output in zip(error_v, final_output_v)]

    # adjust weight & threshold for final output layer
    for i, outer_neuron in enumerate(outer_layer):
        outer_neuron.adjust_weight_and_threshold(error_adjusts[i], hidden_output_v)

    # get adjustment vector for hidden layer
    feedback_ratio_v = [dot(error_adjusts, [neuron.weights[i] for neuron in outer_layer])
                        for i in range(len(hidden_layer))]
    hidden_adjusts = [hidden_output * (1 - hidden_output) * feeback_ratio
                        for hidden_output, feeback_ratio in zip(hidden_output_v, feedback_ratio_v)]
    # print("Feedback ratio {}, hidden_adjusts {}".format(feedback_ratio_v, hidden_adjusts))
    # adjust weight & threshold for hidden layer
    for i, hidden_neuron in enumerate(hidden_layer):
        hidden_neuron.adjust_weight_and_threshold(hidden_adjusts[i], input_v)

def classify(neural_network, input):
    return feed_forward(neural_network, input)[-1]

def network_factory(n_inputs, n_hidden, n_outputs, learning_rate):
    hidden_layer = [Neuron(n_inputs=n_inputs, learning_rate=learning_rate) for _ in range(n_inputs)]
    output_layer = [Neuron(n_inputs=len(hidden_layer), learning_rate=learning_rate) for _ in range(n_outputs)]
    return [hidden_layer, output_layer]

if __name__ == "__main__":
    data = [
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
    ]
    target = [1,1,1,1,0,0,0,0]

    def stepper(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

    alphas = []
    for a in stepper(0.8, 1.6, 0.02):
        network = network_factory(n_inputs=2,n_hidden=2,n_outputs=1, learning_rate=a)
        for i in range(3500):
            error = 0
            for d, t in zip(data, target):
                back_propagate(network, d, t)
                error += (classify(network, d)[0] - t) ** 2
            if error < 0.01:
                alphas.append({'a': a, 'iters': i})
                break

    for alpha in alphas:
        print("Alpha {0} :: {1}".format(alpha['a'], alpha['iters']))
