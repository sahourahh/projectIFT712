
from classifiers.classifier import Classifier
def main ():
    X = [[1, 3, 4], [6, 3, 2]]

    t=[[1,0],[0,1]]

    me = Classifier()

    print(me.train())
    print(me.prediction(X))



if __name__ == '__main__':
    main()