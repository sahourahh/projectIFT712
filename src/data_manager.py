#####
# Aurélien Vauthier (19 126 456)
# Ikram Mekkid (19 143 008)
# Sahar Tahir (19 145 088)
####

import pandas as pd 
import numpy as np
import matplotlib.pyplot
import random 

import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


class DataManager:

    _train_file = "../ressources/leaf-classification/train.csv"
    _images_repo="../ressources/leaf-classification/images"
    _nb_test_data=0.2
    _r=random.randint(17, 27 )
      
             
    def _extractBasicData(self,nb_test_data,r):
        train=pd.read_csv(self._train_file)        
        s = LabelEncoder().fit(train.species)  #len=99 return a list of all species
        classes = list(s.classes_)  # list of all classes,   position=code
        classes_labels = s.transform(train.species)  #len=990 Encode labels with value between 0 and n_classes-1.
        train = train.drop(['species', 'id'], axis=1) 
        sss = StratifiedShuffleSplit(n_splits=1,  test_size=nb_test_data, random_state=r)
        for train_index, test_index in sss.split(train, classes_labels):   #n_splits is the nbre of iterations
            X_train, X_test = train.values[train_index], train.values[test_index]  #train_index:indexes of train data
            y_train, y_test = classes_labels[train_index], classes_labels[test_index]
        return X_train, X_test, y_train, y_test, classes

    def _extractTargets(self,y,classes):
        T=np.zeros([len(y),len(classes)])
        for i in range(0,len(y)):
            T[i,y[i]]=1
        return  T
###########################################################
    def _extractImageData(self,images_repo,file):
        image_data=[]
        data=pd.read_csv(file, sep=',',header=None).values
        data = np.delete(data, 0,axis=0)
        for i in range(0, np.shape(data)[0]):
            b=matplotlib.pyplot.imread(images_repo+'/'+str(data[i][0])+'.jpg', format=None)    
            image_data.append(np.ravel(b))
        return image_data
###########################################################################################################################################
    def getBasicDataTrain(self):      
        return self._extractBasicData(self._nb_test_data,self._r)[0]
    def getBasicDataTest(self):          
        return self._extractBasicData(self._nb_test_data,self._r)[1]
    def getTrainTargets(self):   
        y =self._extractBasicData(self._nb_test_data,self._r)[2]
        classes =self._extractBasicData(self._nb_test_data,self._r)[4]
        return self._extractTargets(y,classes)
    def getTestTargets(self):   
        y =self._extractBasicData(self._nb_test_data,self._r)[3]
        classes =self._extractBasicData(self._nb_test_data,self._r)[4]
        return self._extractTargets(y,classes)
    def getListOfClasses(self):   
        return self._extractBasicData(self._nb_test_data,self._r)[4]
    


    #def getImageTrainData(self):   # return a vector 990
    #    return self._extractImageData(self._images_repo, self._train_file)
    #def getImageTestData(self):   # return a vector 594 
    #    return self._extractImageData(self._images_repo, self._test_file)

    #def getAllTrainData(self):   # return a matrix (990, 193)
    #    return np.c_[self.getBasicDataTrain() , self.getImageTrainData()] 
    #def getAllTestData(self):   # return a matrix (594, 193)
    #    return np.c_[self.getBasicDataTest() , self.getImageTestData()]
    

#dm=DataManager()
#train_file = "../ressources/leaf-classification/train.csv"
#test_file = "../ressources/leaf-classification/test.csv"
#images_repo="../ressources/leaf-classification/images"

#data=dm.getBasicDataTrain()
#y=dm._extractBasicData(train_file,0.2)[2]
#classes=dm._extractBasicData(train_file,0.2)[4]
#print(dm._extractTargets(y,classes))
#print (data.shape)

#print(dm._extractBasicData(train_file,0.2)[1])
    