from classifiers.parametric_classifier import ParametricClassifier

import numpy as np
from sklearn import ensemble, tree

class AdaBoostClassifier(ParametricClassifier):

    def __init__(self, data_manager, useImageData):
        super().__init__(data_manager, useImageData)
        self.model = ensemble.AdaBoostClassifier()
        self.param_grid = {
            "base_estimator" : [tree.DecisionTreeClassifier(max_depth=n) for n in range(13, 14)], # best found : 13
            "n_estimators" : range(83, 89), # best found : 84
            "learning_rate" : np.geomspace(0.01, 1, num=10) # best found : 0.027825
        }