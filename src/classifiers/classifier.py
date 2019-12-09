
class Classifier:
    def __init__(self, DataManager, classifier):
        self.dm = DataManager
        self._clf = classifier
        self.X_train = DataManager.getBasicTrainData()
        self.X_test = DataManager.getBasicTestData()
        self.t_train = DataManager.getTrainTargets()
        self.t_test = DataManager.getTestTargets()

    def train(self):
        """
        train classifier using data manager's training data
        """
        self._clf.fit(self.X_train, self.t_train)

    def prediction(self, X):
        """
        make predictions using new data instance
        :param X: data given
        :return: prediction of data given
        """
        prediction = self._clf.predict(X)
        return prediction

    def _error(self ,X ,t):
        """
        error quantifiying the quality of prediction
        :param X:data input
        :param t:target
        :return: score
        """
        error = self._clf.score(X ,t)
        return error

    def show_stats(self):
        """
        show training error and test error
        """
        print('Training error is {:.4f} %'.format(self._error(self.X_train ,self.t_train) ) *100)
        print('Test error is {:.4f} %'.format(self._error(self.X_test ,self.t_test) ) *100)
