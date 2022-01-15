import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.write('# Can you survived??')
data = pd.read_csv("train.csv")

data = data.drop(columns =['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
data = data.dropna()
st.dataframe(data)
classes = data['Pclass'].unique()
Class= st.sidebar.selectbox('class',(1,2,3))
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
gender = data['Sex'].unique()
Gender = st.sidebar.selectbox('Gender',('0','1'))
data['Age'] = data['Age'].astype(int)
Age = st.sidebar.slider('Age',10,90,20)

x = data.drop(columns='Survived')
y = data['Survived']
model = RandomForestClassifier()
model.fit(x,y)
file = st.sidebar.file_uploader('Upload your input csv file')
if file is not None:
  test = pd.read_csv(file)
else:
  X = {
    'Pclass':Class,
     'Sex':Gender,
     'Age':Age
  }
  test = pd.DataFrame(X,index=[0])
pred = model.predict(test)
st.write("Your Age is" ,str(Age),', You Class is', str(Class),', and You\'re ','Female' if Gender is 1  else 'Male')
if pred==1:
  st.write('# You Can Survive')
else:
  st.write('# You cannot Survive')



