from data_manager import DataManager
from classifiers.SvmRbf import SvmRbf
from classifiers.AdaBoostClassifier import AdaBoostClassifier

def main():
    dm = DataManager()
    classifier = AdaBoostClassifier(dm)
    classifier.train()
    classifier.show_stats()


if __name__ == "__main__":
    main()