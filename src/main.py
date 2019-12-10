from sklearn.linear_model import LogisticRegression
from classifiers.classifier import Classifier
from data_manager import DataManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def main():
    dm = DataManager()

    lr = Classifier(dm, LogisticRegression(solver='liblinear', multi_class='auto'))
    lr.train()
    lr.show_stats()

    rf = Classifier(dm, RandomForestClassifier(n_estimators=10))
    rf.train()
    rf.show_stats()

    mlp = Classifier(dm, MLPClassifier(max_iter=20, warm_start=True, verbose=True, random_state=0))
    mlp.train()
    mlp.show_stats()



if __name__ == "__main__":
    main()