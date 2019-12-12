##################################
# Aurélien Vauthier (19 126 456) #
# Ikram Mekkid (19 143 008)      #
# Sahar Tahir (19 145 088)       #
##################################


import pandas as pd 
import numpy as np
import random 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot

from sklearn.decomposition import PCA
    #####
from PIL import Image
import glob, os
import matplotlib.pyplot
from scipy import ndimage, misc
from tqdm import tqdm
import matplotlib.pyplot as plt 
import cv2
import math
class DataManager:

    _train_file = "../ressources/leaf-classification/train.csv"
    _images_repo = "../ressources/leaf-classification/images/"    
    _r = random.randint(17, 27)
    _X_data_train = pd.DataFrame()
    _X_data_test = pd.DataFrame()
    _X_img_train = pd.DataFrame()
    _X_img_test = pd.DataFrame()
    _X_all_train = pd.DataFrame()
    _X_all_test = pd.DataFrame()
    _y_train = pd.DataFrame()
    _y_test = pd.DataFrame()
    _classes = []
    _id_img_train=[]
    _id_img_test=[]


    def __init__(self,nb_test_data = 0.2, pca=False):
        ###nb_test_data is the percentage of test data from the original file
        self._nb_test_data = nb_test_data
        self._pca = pca

 ########################private_functions            
    def _extractBasicData(self):
        """
        This function generates basic train and test data
        """
        train=pd.read_csv(self._train_file) 
        
        s = LabelEncoder().fit(train.species)  
        self._classes = list(s.classes_)  
        classes_labels = s.transform(train.species)
        id_img= train.id 
        train = train.drop(['species'], axis=1)
        if (self._pca==True):
            trainX = train.drop(['id'], axis=1)
            pca = PCA(n_components=0.85 ,svd_solver='full')
            pca.fit(trainX)
            trainX=pca.transform(trainX)
            train_df=pd.DataFrame.from_records(trainX)
            train_df.insert(loc=0, column='id', value=train['id'])
            train=train_df

        sss = StratifiedShuffleSplit(n_splits=1,  test_size=self._nb_test_data, random_state=self._r)
        for train_index, test_index in sss.split(train, classes_labels):  
            X_train, X_test = train.values[train_index], train.values[test_index]  
            self._y_train, self._y_test = classes_labels[train_index], classes_labels[test_index]

        self._id_img_train =  list(np.int_( X_train[:,0]))
        self._id_img_test =  list(np.int_( X_test[:,0])) 
        self._X_data_train = np.delete(X_train, 0, 1)
        self._X_data_test = np.delete(X_test, 0, 1)

    def _first_left_t (self,matrix):
        """
        This function extracts index of the first white pixel from left to right
        matrix: matrix of the image
        return: index_row :row number
                index_col :column number
        """
        j=0
        while max(matrix[:,j])!=float(1):
            j=j+1
        index_col=j
        i=0
        while matrix[i,j]!=float(1):
            i=i+1
        index_row=i
        return index_row, index_col

    def _first_right_t (self,matrix):
        """
        This function extracts index of the first white pixel from right to left
        matrix: matrix of the image
        return: index_row :row number
                index_col :column number
        """
        j=len(matrix[0,:])-1   
        while max(matrix[:,j])!=float(1):
            j=j-1
        index_col=j
        i=0   
        while matrix[i,j]!=float(1):
            i=i+1
        index_row=i
        return index_row, index_col

    def _first_top_l (self,matrix):  
        """
        This function extracts index of the first white pixel from top to bottom
        matrix: matrix of the image
        return: index_row :row number
                index_col :column number
        """
        i=0   
        while max(matrix[i,:])!=float(1):
            i=i+1
        index_row=i        
        j=0  
        while matrix[i,j]!=float(1):
            j=j+1
        index_col=j
        return index_row, index_col

    def _first_bottom_l (self,matrix):
        """
        This function extracts index of the first white pixel from bottom to top
        matrix: matrix of the image
        return: index_row :row number
                index_col :column number
        """
        i=len(matrix[:,0])-1   
        while max(matrix[i,:])!=float(1):
            i=i-1
        index_row=i        
        j=0   
        while matrix[i,j]!=float(1):
            j=j+1
        index_col=j
        return index_row, index_col

    def _rm_frame(self,image):
        """
        This function removes black frame surrounding the leaf in the image
        image: image object
        return: result :image object
        """        
        image_array=np.asarray(image) #image to array
        left_r,left_c = self._first_left_t (image_array)
        right_r,right_c = self._first_right_t (image_array)
        top_r,top_c = self._first_top_l (image_array)
        bottom_r,bottom_c = self._first_bottom_l (image_array)
        image_array = image_array[top_r:bottom_r+1,left_c:right_c+1]
        result=Image.fromarray(image_array) #array to image
        return result   #return an image
   
    def _blackWhite(self,image):
        """
        This function calculates the percentage of black and white pixels in the image
        image: image object
        return: p0: percentage of black pixels
                p1:percentage of white pixels
        """        
        h = image.histogram()
        nt, n0, n1 = sum(h), h[0], h[-1]
        p0=round(100*n0/nt,2)  #black
        p1=round(100*n1/nt,2)    #white
        return p0, p1
  
    def _ratio_width_length(self,image):
        """
        This function calculates the ratio between width & length
        image: image object
        return: width/length: ratio
        """                
        width, length =image.size  #largeur image[1,:],longueur image[:,1]
        return width/length
  
    def _Contour_Features(self,imagefile):   
        """
        This function calculates different image features based on contour-detection 
        imagefile: image path
        return: peak: nb of peaks of the contour
                eccentricity: eccentricity of the ellipse that fits the contour
                angle: deviation of the ellipse that fits the contour
                m: gradient of the line that fits the contour
                y0: image of the abscissa 0 by the equation of the line that fits the contour
        """       
        peak=0
        original_color  =cv2.imread(imagefile)
        original=cv2.cvtColor(original_color, cv2.COLOR_RGB2GRAY)
        blur=cv2.GaussianBlur(original,(5,5),0)
        ret,thresh=cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh,100,200)
        contours,hierarchy=cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        LENGTH = len(contours)            
        big_cnt=np.vstack([contours[c] for c in range (0,LENGTH)])

        perimetre=cv2.arcLength(big_cnt,True)
        approx = cv2.approxPolyDP(big_cnt,0.01*perimetre,True)
        peak=len(approx)
        (x, y), (MA, ma), angle = cv2.fitEllipse(big_cnt)#angle at which object is directed Major Axis and Minor Axis lengths.
        a = ma / 2
        b = MA / 2
        eccentricity = math.sqrt(pow(a, 2) - pow(b, 2))
        eccentricity = round(eccentricity / a, 2)
        [vx,vy,x,y] = cv2.fitLine(big_cnt, cv2.DIST_L2,0,0.01,0.01) #vx,vy are normalized vector collinear to the line and x0,y0 is a point on the line
        m=vy[0]/vx[0]
        y0=y[0]-m*x[0]
        return peak, eccentricity, angle ,m, y0

    def _extractImagesCaracteristics(self,N):
        """
        This function calculates all image features using the previous methods
        N: list that contains ids of the images
        return: a dataframe of the features
        """   
        image_data= [[0] * 8 for _ in range(len(N))]  # 9 features for each image, nb_images=1584
        for i in tqdm(range(0,len(N))):
            imagefile=self._images_repo+str(N[i])+".jpg"
            image  = Image.open(imagefile)
            image = image.convert('1')
            image=self._rm_frame(image)   
            image_data[i][0], image_data[i][1] = self._blackWhite(image) #percentage of black and white pixels
             
            image_data[i][2] =self._ratio_width_length(image)

            peak, eccentricity, angle ,m ,y0 = self._Contour_Features(imagefile)
            
            image_data[i][3]=peak 
            image_data[i][4]=eccentricity
            image_data[i][5]=angle
            image_data[i][6]=m
            image_data[i][7]=y0
        
        return pd.DataFrame(data=image_data, columns=['black_pxl%', 'white_pxl%', 'ratio_W/L','nb_peak','ellipse_eccentricity','ellipse_deviation','line_gradient','line_y0'])

    def _extractImageData(self):
        """
        This function apply the function _extractImageData to train and test data
        """ 
        if len(self._id_img_train)==0 or len(self._id_img_test)==0  :
            self._extractBasicData() 
        if(len(self._X_img_train)==0 ):
            self._X_img_train=self._extractImagesCaracteristics(self._id_img_train).to_numpy()
        if(len(self._X_img_test)==0):
            self._X_img_test=self._extractImagesCaracteristics(self._id_img_test).to_numpy()       


########################public_functions   
    def getBasicTrainData(self):          
        """
        This function calls the private function _extractBasicData() to extract train data
        :return: _X_data_train: Train matrix
        """
        if len(self._X_data_train)==0 :
            self._extractBasicData() 
        return self._X_data_train   
        
    def getBasicTestData(self):          
        """
        This function calls the private function _extractBasicData() to extract test data
        :return: _X_data_test : Test matrix
        """
        if len(self._X_data_test)==0 :
            self._extractBasicData() 
        return  self._X_data_test

    def getTrainTargets(self):        
        """
        This function calls the private function _extractTrainTargets() to extract train Targets if they aren't already extracted
        :return: A vector of data classes
        """
        if len(self._y_train)==0 :
            self._extractBasicData()   
        return self._y_train

    def getTestTargets(self):
        """
        This function calls the private function _extractTestTargets() to extract test Targets if they aren't already extracted
        :return: A vector of data classes
        """  
        if len(self._y_test)==0 :
            self._extractBasicData()      
        return self._y_test

    def getListOfClasses(self):
        """
        This function  lists all the classes
        :return: vector of all classes
        """ 
        if len(self._classes)==0 :
            self._extractBasicData()   
        return self._classes

    def getImageTrainData(self):          
        """
        This function extract image features for train data
        :return: _X_img_train : Train image features matrix
        """
        if len(self._X_img_train)==0 :
            self._extractImageData()        
        return self._X_img_train
 
    def getImageTestData(self):          
        """
        This function extract image features for test data
        :return: _X_img_test  : Test image features matrix
        """
        if len(self._X_img_test)==0:
            self._extractImageData()        
        return self._X_img_test

    def getALLTrainData(self):          
        """
        This function merge basic train data with image features
        :return: _X_train: Train matrix
        """
        if len(self._X_all_train)==0:
            if len(self._X_img_train)==0 :
                self._extractImageData() 
            self._X_all_train=np.concatenate((self._X_data_train, self._X_img_train), axis=1)
        return self._X_all_train
    
    def getALLTestData(self):          
        """
        This function merge basic train data with image features
        :return: _X_test: Test matrix
        """
        if len(self._X_all_test)==0:
            if len(self._X_img_test)==0 :
                self._extractImageData() 
            self._X_all_test=np.concatenate((self._X_data_test, self._X_img_test), axis=1)
        return self._X_all_test
