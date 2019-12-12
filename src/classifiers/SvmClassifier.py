from classifiers.parametric_classifier import ParametricClassifier

import numpy as np
from sklearn.svm import SVC

class SvmClassifier(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = SVC()
        self.param_grid = {
            "kernel" : ['poly'],# 'rbf', 'sigmoid'],    # best found : 'poly'
            "C" : np.geomspace(0.0001, 0.001, num=5),   # best found : 0.00031622776601683794
            "gamma": np.geomspace(0.0001, 0.1, num=4),  # best found : 0.0001 (tends to infinitely loop bellow)
            "coef0": np.linspace(5, 25, num=5),    # best found : 20.0
            "degree": range(7, 11)  # best found : 8
        }

    def train(self, verbose=True):
        """
        train override to search for the best hyper parameters before actually training the model with the probability
        attribute set to True. It is not set in the __init__ in order to accelerate the cross validation.
        WARNING : this method might take a long time to execute
        :param verbose: whether to show verbosity for cross validation or not
        :return: None, the model is considered trained with the best hyper parameters found once this method is finished
        """
        super().train(verbose=verbose)
        self._clf.probability = True
        self._clf.fit(self.X_train, self.t_train)


