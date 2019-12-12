from classifiers.parametric_classifier import ParametricClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LinearDiscriminantAnalysisClassifier(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = LinearDiscriminantAnalysis()
        self.param_grid = {
            "solver": ('lsqr', 'eigen') # the 'svd' solver can not be use because of collinear data
        }