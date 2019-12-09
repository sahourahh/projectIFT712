

from sklearn.linear_model import LogisticRegression
from classifiers.classifier import Classifier


def main():

    Classifier.train(LogisticRegression)

    Classifier.prediction(LogisticRegression, Classifier.X_test)

    Classifier._error(LogisticRegression, Classifier.X_test, Classifier.t_test)

    Classifier.show_stats(LogisticRegression)


if __name__ == "__main__":
    main()
