from sklearn.ensemble import RandomForestClassifier
from classifiers.classifier import Classifier
class RandomForest:
def __init__(self,  classifier):
    self.clf = RandomForestClassifier

Rfc= Classifier.train(RandomForestClassifier)

Rf_pred=Classifier.prediction(RandomForestClassifier,Classifier.X_test)

Rf_error= Classifier._error(RandomForestClassifier,Classifier.X_test,Classifier.t_test)
