# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#Loading the saved model
loaded_model=pickle.load(open('C:/project/ML/trained_models/heart_data_trained_model.sav','rb'))

input_data=(62,0,0,140,268,0,0,160,0,3.6,0,2,2)
#Change the input to a numpy array

input_data_as_numpy_array=np.asarray(input_data)

#Reshape the numpy array as we are predicting the target for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)

if (prediction[0]==0):
    print('The person does not have a heart disease')
else:
    print('The person has a heart disease')