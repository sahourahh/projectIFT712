from sklearn.linear_model import LogisticRegression
from classifiers.parametric_classifier import ParametricClassifier

class LogisticRegressionClassifier(ParametricClassifier):

    def __init__(self, data_manager, useImageData):
        super().__init__(data_manager, useImageData)
        self.model=LogisticRegression(solver='liblinear', multi_class='auto', max_iter=1000)
        self.param_grid = {
            'penalty' : ['l1','l2']
        }