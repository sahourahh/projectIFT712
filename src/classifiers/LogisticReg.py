from sklearn.linear_model import LogisticRegression
from classifiers.parametric_classifier import ParametricClassifier
class Logisticregression(ParametricClassifier):

    def __init__(self,data_manager):

        super().__init__(data_manager)
        self.model=LogisticRegression(solver='liblinear',multi_class='auto')
        self.param_grid={'penalty':['l1','l2']}