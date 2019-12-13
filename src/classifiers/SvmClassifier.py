from classifiers.parametric_classifier import ParametricClassifier

import numpy as np
from sklearn.svm import SVC

class SvmClassifier(ParametricClassifier):

    def __init__(self, data_manager, useImageData):
        super().__init__(data_manager, useImageData)
        self.model = SVC()
        self.param_grid = {
            "kernel" : ['rbf', 'sigmoid'],# 'poly'],
            "C" : np.geomspace(0.00001, 1, num=6),
            "gamma": np.geomspace(0.00001, 0.01, num=4),  # WARNING : tends to infinitely loop bellow 0.0001 for 'poly'
            "coef0": np.linspace(20, 15, num=3),
            # "degree": range(2, 7, 2)  # not used because we deactivated the 'poly' kernel
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


