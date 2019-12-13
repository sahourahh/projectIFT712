import numpy as np
from sklearn.neural_network import MLPClassifier
from classifiers.parametric_classifier import ParametricClassifier

class Neuralnetwork(ParametricClassifier):

    def __init__(self, data_manager, useImageData):
        super().__init__(data_manager, useImageData)
        self.model = MLPClassifier(max_iter=1000)
        self.param_grid = {
            'learning_rate_init': np.linspace(0.1, 1, num=2),
            'activation': ['identity'],  # 'relu', 'tanh'],
            'solver': ['adam'],  # 'sgd', 'lbfgs'],
        }

