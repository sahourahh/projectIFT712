from sklearn.neural_network import MLPClassifier
from classifiers.parametric_classifier import ParametricClassifier
import numpy as np
from itertools import product
class Neuralnetwork(ParametricClassifier):

    def __init__(self, data_manager,useImageData):
        super().__init__(data_manager,useImageData)
        self.model = MLPClassifier(max_iter=100,warm_start=0)

        self.param_grid = {'learning_rate_init': np.linspace(start=0.01, stop=0.5, num=20),  # best=0.01
                            'activation': ['tanh', 'relu', 'identity'],  # best=identity
                            'solver': ['sgd', 'adam', 'lbfgs'],  # best=adam
                           }