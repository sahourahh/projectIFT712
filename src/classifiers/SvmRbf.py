from classifiers.parametric_classifier import ParametricClassifier

import numpy as np
from sklearn.svm import SVC

class SvmRbf(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = SVC(kernel='rbf')
        self.param_grid = {"gamma": np.linspace(1, 5, num=25)}