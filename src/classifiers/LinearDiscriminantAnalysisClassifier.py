from classifiers.parametric_classifier import ParametricClassifier

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LinearDiscriminantAnalysisClassifier(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = LinearDiscriminantAnalysis()
        self.param_grid = {
            # the 'svd' solver can not be use because of collinear data
            "solver": ('lsqr', 'eigen'),    # best found : 'lsqr'
            # furthermore, the shrinkage doesn't work with the 'svd' solver
            "shrinkage": np.geomspace(0.000000001, 1).tolist() + [None, 'auto'] # best found : 4.71486636345739e-06
        }