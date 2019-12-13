from data_manager import DataManager
from classifiers.AdaBoostClassifier import AdaBoostClassifier
from classifiers.LinearDiscriminantAnalysisClassifier import LinearDiscriminantAnalysisClassifier
from classifiers.LogisticReg import LogisticRegressionClassifier
from classifiers.NeuralNetwork import Neuralnetwork
from classifiers.RandomForest import RandomForest
from classifiers.SvmClassifier import SvmClassifier

def main():
    # list of all the classifiers
    classifiers = [LogisticRegressionClassifier, SvmClassifier, RandomForest,
                   AdaBoostClassifier, Neuralnetwork, LinearDiscriminantAnalysisClassifier]

    # use one data manager for all classifiers
    data_manager = DataManager(pca=False)

    for classifier_class in classifiers:
        # init classifier
        classifier = classifier_class(data_manager, useImageData=True)

        # in order to run faster, the number of cross-validation params has been severely reduced
        classifier.train(verbose=True)  # the neural network's classifier produces Convergence Warnings

        # print classifier's accuracy and cross entropy loss
        classifier.show_stats()


if __name__ == "__main__":
    main()