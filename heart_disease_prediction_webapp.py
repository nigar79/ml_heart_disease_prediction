# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 18:04:06 2022

@author: Nigar S
"""

import numpy as np
import pickle
import streamlit as st


#Loading the saved model
loaded_model=pickle.load(open('C:/project/ML/heart_disease_prediction/trained_model.sav','rb'))

#Prediction function

def heart_defect_prediction(input_data):
    
    #Change the input to a numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #Reshape the numpy array as we are predicting the target for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)

    if (prediction[0]==0):
        return 'The person does not have a heart disease'
    else:
        return 'The person has a heart disease'
    
    
    
def main():
    
    #Giving a title for our webpage 
    st.title('Heart Defect Prediction Web APP')
    
    age=st.text_input('Age of a Person')
    sex=st.text_input('Gender')
    cp=st.text_input('Chest Pain Type')
    trestbps=st.text_input('The person\'s resting blood pressure in mm/Hg')
    chol=st.text_input('Serum Cholesterol in mg/dl')
    fbs=st.text_input('The person\'s fasting blood sugar in mg/dl')
    restecg=st.text_input('resting electrocardiographic results')
    thalach=st.text_input('Maximum heart rate achieved')
    exang=st.text_input('Exercise Induced Angina')
    oldpeak=st.text_input('ST depression induced by exercise relative to rest')
    slope=st.text_input('ST segment shift relative to exercise-induced')
    ca=st.text_input('number of major vessels')
    thal=st.text_input('Thalassemia(blood disorder)')
        
    #Code for Prediction
    diagnosis=''
    
    #Creating button for Prediction
    
    if st.button('heart disease test result'):
        diagnosis=heart_defect_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    

    
    
    