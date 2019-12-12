from sklearn.ensemble import RandomForestClassifier
from classifiers.parametric_classifier import ParametricClassifier
import numpy as np
class RandomForest(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = RandomForestClassifier()
        self.param_grid = {"max_depth": np.linspace(10, 100, num = 10),   #best=60
                           "n_estimators":range(50,100) }   #   best=99