from sklearn.linear_model import LogisticRegression
from classifiers.classifier import Classifier
from data_manager import DataManager
from classifiers.RandomForest import Randomforest
from classifiers.NeurolNetwork import Neurolnetwork
def main():
    dm = DataManager()

    lr = Classifier(dm, LogisticRegression(solver='liblinear', multi_class='auto'))
    lr.train()
    lr.show_stats()

    rf = Randomforest(dm)
    rf.train()
    rf.show_stats()

    mlp = Neurolnetwork(dm)
    mlp.train()
    mlp.show_stats()



if __name__ == "__main__":
    main()