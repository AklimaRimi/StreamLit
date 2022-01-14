import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import cv2 as cv

#img = cv.read

st.write(
    """
    # Titanic Dataset
    The Images and model build for Titanic dataset
    """
)
img = plt.imread("C:/Users/user/Downloads/titanic.jpg")
st.image(img,use_column_width = True)
data= pd.read_csv("F:/dataset/titanic/train.csv")
st.header('Enter Gender and Age')
input = "female"
input2 ='45'
gender = st.text_area('Input',input,height=1)
age = st.text_area('Age',input2,height=1)

st.write("""
***
""")

st.write("""
* You have entered 
""")
'Gender = '+ gender+ ' and Age = '+ age

x = data[data['Sex']==gender]
st.write("""
# Survived data
""")

n=x['Survived'].value_counts()
n
st.write("""
# Visualization the percentage of survived rate
""")
fig = plt.figure(figsize=(1,1))
sns.histplot(x['Survived'])
st.pyplot(fig)
