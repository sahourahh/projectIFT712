# -*- coding: utf-8 -*-

from classifiers.classifier import Classifier
from sklearn.model_selection import GridSearchCV


class ParametricClassifier(Classifier):
    """
    Parent class for any Classifier with hyper parameters
    """

    def __init__(self, data_manager, k=3):
        """
        Init any Classifier with hyper parameters
        :param data_manager: data manager used by the classifier
        :param k: number of cross-validation's folds
        """
        super().__init__(data_manager)
        self.k = k  # nb of cross-validation loops

    def train(self):
        """
        train override to search for the best hyper parameters before actually training the model
        WARNING : this method might take a long time to execute
        :return: None, the model is considered trained with the best hyper parameters found once this method is finished
        """
        try:
            grid_search = GridSearchCV(self.model, self.param_grid, n_jobs=-1, cv=self.k)
        except AttributeError:
            raise AttributeError("Please set ParametricClassifier's subclass self.model and self.param_grid before"
                                 " trying to train the classifier.")

        self._clf = grid_search.fit(self.X_train, self.y_train).best_estimator_
        super().train()
