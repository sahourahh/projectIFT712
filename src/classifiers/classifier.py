from sklearn.metrics import accuracy_score, log_loss

class Classifier:
    def __init__(self, DataManager, classifier, useImageData=False):
        self.dm = DataManager
        self._clf = classifier

        if useImageData:
            self.X_train = DataManager.getALLTrainData()
            self.X_test = DataManager.getALLTestData()
        else:
            self.X_train = DataManager.getBasicTrainData()
            self.X_test = DataManager.getBasicTestData()

        self.t_train = DataManager.getTrainTargets()
        self.t_test = DataManager.getTestTargets()

    def train(self):
        """
        train classifier using data manager's training data
        """
        self._clf.fit(self.X_train, self.t_train)

    def prediction(self, X, predict_proba=False):
        """
        make predictions using new data instance
        :param X: data given
        :return: prediction of data given
        """
        if predict_proba:
            prediction = self._clf.predict_proba(X)
        else:
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

    def loss(self, X, t):
        """
        Compute and return the cross entropy loss (or log loss) for the given X data and it's true targets t
        :param X: the data to predict
        :param t: the true data's targets
        :return: the classifier's cross entropy loss on the given data
        """
        return log_loss(t, self.prediction(X, predict_proba=True))

    def accuracy(self, X, t):
        """
        Compute and return the accuracy for the given X data and it's true targets t
        :param X: the data to predict
        :param t: the true data's targets
        :return: the classifier's accuracy on the given data
        """
        return accuracy_score(t, self.prediction(X))

    def show_stats(self):
        """
        print training and test accuracy and loss
        """
        # print classifier's class name
        print("--- {} ---".format(self._clf.__class__.__name__))

        # print training's accuracy and loss
        print('Training results : accuracy is {:.4%}\tloss is {:.4f}'.format(
            self.accuracy(self.X_train, self.t_train), self.loss(self.X_train, self.t_train) ))

        # print test's accuracy and loss
        print('Test results     : accuracy is {:.4%}\tloss is {:.4f}'.format(
            self.accuracy(self.X_test, self.t_test), self.loss(self.X_test, self.t_test) ))

        # print new line for readability
        print()
