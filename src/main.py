from sklearn.linear_model import LogisticRegression
from classifiers.classifier import Classifier
from data_manager import DataManager


def main():

    dm=DataManager()
    classifier=Classifier(dm, LogisticRegression)

    classifier.train()

    classifier._error()

    classifier.show_stats()


if __name__ == "__main__":
    main()