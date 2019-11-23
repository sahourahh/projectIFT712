
import numpy as np
from data_manager import DataManager

def main():

    dm=DataManager()
    dataextraction=dm.getBasicTrainTestData()    
    print ("Size of train data:",dataextraction[0].shape)         #(792, 192)
    print ("Train data:\n",dataextraction[0])              
    
    print ("Size of data test:",dataextraction[1].shape)      #(198, 192)
    print ("Data test:\n",dataextraction[1])
    
    print ("Size Train tragets:\n",dm.getTrainTargets().shape)
    print ("Train tragets:\n",dm.getTrainTargets())
    
    print ("Size Test targets:\n",dm.getTestTargets().shape)
    print ("Test targets:\n",dm.getTestTargets())
    
    print ("Size classes:\n",len(dm.getListOfClasses()))
    print ("classes:\n",dm.getListOfClasses())

if __name__ == "__main__":
    main()