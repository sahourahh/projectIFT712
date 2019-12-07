
from sklearn.linear_model import LinearRegression


class Classifier:
    def __init__(self, DataManager):
        self.dm = DataManager
        self._clf = LinearRegression()
        self.X_train = DataManager.getBasicTrainTestData()[0]
        self.X_test = DataManager.getBasicTrainTestData()[1]
        self.t_train = DataManager.getTrainTargets()
        self.t_test = DataManager.getTestTargets()

    def train(self):
        """
        train classifier using data manager's training data

        """
        classifier = self._clf.fit(self.X_train, self.t_train)


    def prediction(self, X):
        """
        make predictions using new data instance
        :param X: data given
        :return: prediction of data given
        """
        prediction = self._clf.predict(X)
        return prediction

    def _error(self,X,t):
        """
        error quantifiying the quality of prediction
        :param X:data input
        :param t:target
        :return: score
        """
        error = self._clf.score(X,t)
        return error
        
    def show_stats(self):
        """
        show training error and test error
        :return: a message indicating our training error and test error
        """
        print('training error is {}'.format(self._error(self.X_train,self.t_train)))
        print('test error is {}'.format(self._error(self.X_test,self.t_test)))




