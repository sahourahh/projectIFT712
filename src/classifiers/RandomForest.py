from sklearn.ensemble import RandomForestClassifier
from classifiers.parametric_classifier import ParametricClassifier
import numpy as np
class Randomforest(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = RandomForestClassifier(n_estimators=10)
        self.param_grid = {"max_depth": np.linspace(10, 100, num = 10)}