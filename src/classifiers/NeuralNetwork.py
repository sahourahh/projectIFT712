from sklearn.neural_network import MLPClassifier
from classifiers.parametric_classifier import ParametricClassifier
import numpy as np
class Neuralnetwork(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = MLPClassifier(max_iter=10,warm_start=0)

        self.param_grid = {'hidden_layer_sizes': list(product(range(100, 120), range(15, 20))),   # best= (104, 19)
                            'learning_rate_init': np.linspace(start=0.01, stop=0.2, num=10),  # best=0.052222222222222225
                            'activation': ['tanh', 'relu', 'identity'],  # best=identity
                            'solver': ['sgd', 'adam', 'lbfgs'],  # best=adam
                           }