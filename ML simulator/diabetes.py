import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import quantile_transform




st.header('**Dataset** ')
if 'drop' not in st.session_state:
  st.session_state['drop'] = []

if 'vizmethod' not in st.session_state:
  st.session_state['vizmethod'] = []


if 'preprocess' not in st.session_state:
  st.session_state['preprocess'] =[]

if 'cleaneddata' not in st.session_state:
  st.session_state['cleaneddata'] = []

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
   # Can be used wherever a "file-like" object is accepted:
   data = pd.read_csv(uploaded_file)
   st.write(data)
else:
  data = pd.read_csv('diabetes.csv')
  st.dataframe(data.head())

def dataclean(data):  
  data = data.dropna()
  cols = data.columns
  cols = cols.to_list()
  drop1,drop2 = st.sidebar.columns(2)
  dropcols = drop1.multiselect('Drop Columns',cols,cols[1])
  if drop2.button('Drop') or st.session_state['drop']:
    st.session_state['drop'] = dropcols
    data = data.drop(columns = st.session_state['drop'])
    st.write('After Dropping New data is')
    st.dataframe(data.head())
    if len(dropcols) ==1:
      st.sidebar.write('Column is droped')
    else:
      st.sidebar.write('Columns are droped')
    
  return data

cleandata = dataclean(data)
st.session_state['cleaneddata']= cleandata

if 'vizcolumns' not in st.session_state:
  st.session_state['vizcolumns'] = cleandata

## ----------------------viz-------##

 
st.sidebar.write('Visualizing Section')

viz = ('histplot','distplot','boxplot')

cols = cleandata.columns
cols = cols.to_list()


heat = st.button('Click here to see relation between columns')
if heat:
  heat = plt.figure(figsize=(10,4))
  sns.heatmap(cleandata.corr(),fmt='.1f',annot=True,mask = np.triu(cleandata.corr()))
  st.pyplot(heat)
def dataviz(cleandata):
  
  vizcolumns = st.sidebar.multiselect('Select column for Visualizing',cols,cols[1])
  vizu1,vizu2 = st.sidebar.columns(2)
  vizmethod = vizu1.selectbox('Select for Visualizing ',viz)
  
 
  if vizu2.button('Show') or st.session_state['vizmethod']:
    st.session_state['vizmethod'] = vizmethod
    st.session_state['vizcolumns'] = vizcolumns
    if vizmethod == 'histplot':  
      for i in vizcolumns:
        hist = plt.figure(figsize=(15,4))
        sns.histplot(st.session_state['cleaneddata'][i])
        st.pyplot(hist)
    elif vizmethod == 'distplot':  
      for i in (vizcolumns):
        dist = plt.figure(figsize=(15,4))
        sns.distplot(st.session_state['cleaneddata'][i])
        st.pyplot(dist)
    elif vizmethod == 'boxplot':  
      for i in (vizcolumns):
        box = plt.figure(figsize=(15,4))
        sns.boxplot(st.session_state['cleaneddata'][i])
        st.pyplot(box)        
dataviz(cleandata)

##------------clean--------##



st.sidebar.write('Clean outlier')

clean1,clean2 = st.sidebar.columns(2)
cols = cleandata.columns.to_list()
col = clean1.multiselect('',cols,cols[1])


def remove(data):
  Q1 = np.percentile(data, 25, interpolation = 'midpoint')
  
# Third quartile (Q3)
  Q3 = np.percentile(data, 75, interpolation = 'midpoint')
  iqr = Q3 - Q1
  percentile25 = data.quantile(0.25)
  percentile75 = data.quantile(0.75)

  upper_limit = percentile75 + 1.5 * iqr
  lower_limit = percentile25 - 1.5 * iqr
  data = np.where(
    data > upper_limit,upper_limit,
    np.where(
        data < lower_limit,
        lower_limit,
        data
    )
  )
  return data


new_data  = cleandata
if clean2.button('Clean Outlier') :
  st.write('Outlier is cleaned')
  st.session_state['preprocess']  = col
  for i in st.session_state['preprocess'] :
    new_data[i] = remove(cleandata[i])
    st.session_state['cleaneddata'] = new_data
    box = plt.figure(figsize=(15,4))
    sns.boxplot(st.session_state['cleaneddata'][i])
    st.pyplot(box) 



##----skewed----##
st.sidebar.write('Skewed data?\n Want to Transform?')

skew1,skew2 = st.sidebar.columns(2)
skew = skew1.multiselect('Select column for Transform',(st.session_state['cleaneddata'].columns).to_list(),(st.session_state['cleaneddata'].columns).to_list()[1])



if skew2.button('Transform'):
  st.write('Data is Transformed')
  for i in skew:
    st.session_state['cleaneddata'][i] = quantile_transform(st.session_state['cleaneddata'][[i]],output_distribution='normal')
    hist = plt.figure(figsize=(15,4))
    sns.distplot(st.session_state['cleaneddata'][i])
    st.pyplot(hist)

## -------- train -------

st.write('#Time to Train model')
model1,model2,model3 = st.columns(3)
smodel = ('Classifier','Regression')

s = model1.selectbox('Choose',smodel)
s2 = model2.selectbox('Which column will be trained',st.session_state['cleaneddata'].columns)
models = ('DecisionTree','RandomForest','XGBoost','CatBoost')
x  = st.session_state['cleaneddata'].drop(columns=s2)
y = st.session_state['cleaneddata'][s2]

from sklearn.model_selection import train_test_split

trainx,testx,trainy,testy = train_test_split(x,y,test_size=.2)


s3 = model3.selectbox('Select model',models)

_x,button1,_xx = st.columns(3)

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor 
from catboost import CatBoostClassifier, CatBoostRegressor
if button1.button('train'):
  if s=='Classifier':
    if s3 == 'DecisionTree':
      
      model = DecisionTreeClassifier()
      model.fit(trainx,trainy)
      pred = model.predict(testx)
      testy = np.array(testy)
      pred = np.array(pred)

      see = plt.figure(figsize=(15,6))
      plt.plot(pred,color='red',)
      plt.plot(testy,color='green',)
      plt.legend(['predict','Actual'])
      st.pyplot(see)
      
      st.header('Accuracy is :'+str(model.score(testx,testy)))
    elif s3 == 'XGBoost':

      model = XGBClassifier()
      model.fit(trainx,trainy)
      pred = model.predict(testx)
      testy = np.array(testy)
      pred = np.array(pred)

      see = plt.figure(figsize=(15,6))
      plt.plot(pred,color='red',)
      plt.plot(testy,color='green',)
      plt.legend(['predict','Actual'])
      st.pyplot(see)
      st.header('Accuracy is :'+str(model.score(testx,testy)))
    elif s3 == 'RandomForest':

      model = RandomForestClassifier()
      model.fit(trainx,trainy)
      pred = model.predict(testx)
      testy = np.array(testy)
      pred = np.array(pred)

      see = plt.figure(figsize=(15,6))
      plt.plot(pred,color='red',)
      plt.plot(testy,color='green',)
      plt.legend(['predict','Actual'])
      st.pyplot(see)
      st.header('Accuracy is :'+str(model.score(testx,testy)))
    elif s3 == 'CatBoost':

      model = CatBoostClassifier()
      model.fit(trainx,trainy)
      pred = model.predict(testx)
      testy = np.array(testy)
      pred = np.array(pred)

      see = plt.figure(figsize=(15,6))
      plt.plot(pred,color='red',)
      plt.plot(testy,color='green',)
      plt.legend(['predict','Actual'])
      st.pyplot(see)
      st.header('Accuracy is :'+str(model.score(testx,testy)))
  if s=='Regression':
    if s3 == 'DecisionTree':
      
      model = DecisionTreeRegressor()
      model.fit(trainx,trainy)
      pred = model.predict(testx)
      testy = np.array(testy)
      pred = np.array(pred)

      see = plt.figure(figsize=(15,6))
      plt.plot(pred,color='red',)
      plt.plot(testy,color='green',)
      plt.legend(['predict','Actual'])
      st.pyplot(see)
      
      st.header('Accuracy is :'+str(model.score(testx,testy)))
    elif s3 == 'XGBoost':

      model = XGBRegressor()
      model.fit(trainx,trainy)
      pred = model.predict(testx)
      testy = np.array(testy)
      pred = np.array(pred)

      see = plt.figure(figsize=(15,6))
      plt.plot(pred,color='red',)
      plt.plot(testy,color='green',)
      plt.legend(['predict','Actual'])
      st.pyplot(see)
      st.header('Accuracy is :'+str(model.score(testx,testy)))
    elif s3 == 'RandomForest':

      model = RandomForestRegressor()
      model.fit(trainx,trainy)
      pred = model.predict(testx)
      testy = np.array(testy)
      pred = np.array(pred)

      see = plt.figure(figsize=(15,6))
      plt.plot(pred,color='red',)
      plt.plot(testy,color='green',)
      plt.legend(['predict','Actual'])
      st.pyplot(see)
      st.header('Accuracy is :'+str(model.score(testx,testy)))
    elif s3 == 'CatBoost':

      model = CatBoostRegressor()
      model.fit(trainx,trainy)
      pred = model.predict(testx)
      testy = np.array(testy)
      pred = np.array(pred)

      see = plt.figure(figsize=(15,6))
      plt.plot(pred,color='red',)
      plt.plot(testy,color='green',)
      plt.legend(['predict','Actual'])
      st.pyplot(see)
      st.header('Accuracy is :'+str(model.score(testx,testy)))








