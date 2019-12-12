from data_manager import DataManager
from classifiers.RandomForest import RandomForest

from sklearn.svm import SVC

def main():
    dm=DataManager(pca=True)
    classifier = RandomForest(dm,useImageData=True)
    classifier.train(verbose=True)
    classifier.show_stats()


if __name__== "__main__":
    main()