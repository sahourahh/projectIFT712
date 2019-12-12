from data_manager import DataManager
from classifiers.SvmRbf import SvmRbf

from sklearn.svm import SVC

def main():
    dm=DataManager()
    classifier = SvmRbf(dm)
    classifier.train()
    classifier.show_stats()


if __name__ == "__main__":
    main()