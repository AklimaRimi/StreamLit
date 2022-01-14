import streamlit as st
import pandas as pd
import numpy as np
import base64
path="F:/dataset/titanic/train.csv"

st.write("""
\t \t # Titanic Filtered data 
""")

st.sidebar.header('Parameters')
seletct_gender = st.sidebar.selectbox('Gender',['Female','Male'])
#data = pd.read_csv(path)
#data
@st.cache
def _data(gender):
    data = pd.read_csv(path)
    data = data.dropna()
    data = data.drop(columns=['PassengerId'])
    return data
data = _data(seletct_gender)
em = ['C','S','Q']
select_cabin = st.sidebar.multiselect('Embarked',em,em)
sur = [0,1]
select_survived = st.sidebar.multiselect('Survived',sur,sur)

#select_cabin,select_survived,seletct_gender

data = data[(data['Survived'].isin(select_survived)) & (data['Sex']==(seletct_gender).lower()) & (data['Embarked'].isin(select_cabin))]
st.dataframe(data)
st.write("""
# After Filtering
""")
'Data size :' + str(data.shape[0])



