##################################
# Aurélien Vauthier (19 126 456) #
# Ikram Mekkid (19 143 008)      #
# Sahar Tahir (19 145 088)       #
##################################


from PIL import Image
import imagehash  #pip install ImageHash
import glob, os
import matplotlib.pyplot

from scipy import ndimage, misc
####################

import pandas as pd 
import numpy as np
import random 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot


class DataManager:

    _train_file = "../ressources/leaf-classification/train.csv"
    _images_repo = "../ressources/leaf-classification/images/"    
    _r = random.randint(17, 27)
    _X_train = pd.DataFrame()
    _X_test = pd.DataFrame()
    _y_train = pd.DataFrame()
    _y_test = pd.DataFrame()
    _classes = []
    _imageFeature=[]



    def __init__(self,nb_test_data = 0.2):
        ###nb_test_data is the percentage of test data from the original file
        self._nb_test_data = nb_test_data
             
    def _extractBasicData(self):
        """
        This function generates train and test data
        """
        train=pd.read_csv(self._train_file)        
        s = LabelEncoder().fit(train.species)  
        self._classes = list(s.classes_)  
        classes_labels = s.transform(train.species) 
        train = train.drop(['species', 'id'], axis=1) 
        sss = StratifiedShuffleSplit(n_splits=1,  test_size=self._nb_test_data, random_state=self._r)
        for train_index, test_index in sss.split(train, classes_labels):  
            self._X_train, self._X_test = train.values[train_index], train.values[test_index]  
            self._y_train, self._y_test = classes_labels[train_index], classes_labels[test_index]

    


###########################################################################################################################################
    
    def getBasicTrainData(self):          
        """
        This function calls the private function _extractBasicData() to extract train data
        :return: _X_train: Train matrix
        """
        if len(self._X_train)==0 :
            self._extractBasicData() 
        return self._X_train
        
    def getBasicTestData(self):          
        """
        This function calls the private function _extractBasicData() to extract test data
        :return: _X_test : Test matrix
        """
        if len(self._X_test)==0 :
            self._extractBasicData() 
        return  self._X_test 


    def getTrainTargets(self):        
        """
        This function calls the private function _extractTrainTargets() to extract train Targets if they aren't already extracted
        :return: A vector of data classes
        """
        if len(self._y_train)==0 :
            self._extractTrainTargets()  
        return self._y_train

    def getTestTargets(self):
        """
        This function calls the private function _extractTestTargets() to extract test Targets if they aren't already extracted
        :return: A vector of data classes
        """  
        if len(self._y_test)==0 :
            self._extractTestTargets()      
        return self._y_test

    def getListOfClasses(self):
        """
        This function  lists all the classes
        :return: vector of all classes
        """ 
        if len(self._classes)==0 :
            self._extractBasicData()   
        return self._classes

