#####
# Aurélien Vauthier (19 126 456)
# Ikram Mekkid (19 143 008)
# Sahar Tahir (19 145 088)
####

import pandas as pd 
import numpy as np
import random 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot

class DataManager:

    _train_file = "../ressources/leaf-classification/train.csv"
    _images_repo = "../ressources/leaf-classification/images"
    _nb_test_data = 0.2
    _r = random.randint(17, 27)
    _X_train = pd.DataFrame()
    _X_test = pd.DataFrame()
    _y_train = pd.DataFrame()
    _y_test = pd.DataFrame()
    _classes = []
             
    def _extractBasicData(self,nb_test_data,r):
        train=pd.read_csv(self._train_file)        
        s = LabelEncoder().fit(train.species)  #len=99 return a list of all species
        self._classes = list(s.classes_)  # list of all classes,   position=code
        classes_labels = s.transform(train.species)  #len=990 Encode labels with value between 0 and n_classes-1.
        train = train.drop(['species', 'id'], axis=1) 
        sss = StratifiedShuffleSplit(n_splits=1,  test_size=nb_test_data, random_state=r)
        for train_index, test_index in sss.split(train, classes_labels):   #n_splits is the nbre of iterations
            self._X_train, self._X_test = train.values[train_index], train.values[test_index]  #train_index:indexes of train data
            self._y_train, self._y_test = classes_labels[train_index], classes_labels[test_index]

    def _extractTargets(self,y,classes):
        T=np.zeros([len(y),len(classes)])
        for i in range(0,len(y)):
            T[i,y[i]]=1
        return  T

###########################################################################################################################################
    def getBasicTrainTestData(self):       ## returns 2 matrix: Xtrain==getBasicTrainTestData()[0] & Xtest====getBasicTrainTestData()[1]    
        
        self._extractBasicData(self._nb_test_data,self._r) 
        return self._X_train , self._X_test 

    def getTrainTargets(self):   
        
        if len(self._classes)==0 or self._y_train.shape==(0,0):
            self._extractBasicData(self._nb_test_data,self._r)
        return self._extractTargets(self._y_train,self._classes)

    def getTestTargets(self):   
        
        if len(self._classes)==0 or self._y_test.shape==(0,0):
            self._extractBasicData(self._nb_test_data,self._r)
        return self._extractTargets(self._y_test,self._classes)

    def getListOfClasses(self):
        
        if len(self._classes)==0 :
            self._extractBasicData(self._nb_test_data,self._r)   
        return self._classes
    


    