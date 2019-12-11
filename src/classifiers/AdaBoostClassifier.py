from classifiers.parametric_classifier import ParametricClassifier

import numpy as np
from sklearn import ensemble, tree

class AdaBoostClassifier(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = ensemble.AdaBoostClassifier()
        self.param_grid = {
            "base_estimator" : [tree.DecisionTreeClassifier(max_depth=n) for n in range(5, 6)], # best found : 5
            "n_estimators" : range(50, 65), # best found : 57
            "learning_rate" : np.geomspace(0.01, 1, num=15) # best found : 0.02131663116533842
        }