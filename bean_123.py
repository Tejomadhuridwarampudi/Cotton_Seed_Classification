import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import math
st.markdown(f'<h1 style="color:BLUE;font-size:30px;">{"BLACK EYED PEAS - DRY BEAN CLASSIFICATION"}</h1>', unsafe_allow_html=True)
#st.title("Black-Eyed Peas - DRY BEAN CLASSIFICATION")
import streamlit as st
st.image("a.jpg")



import pandas as pd
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()




path = st.sidebar.file_uploader("Black-Eyed Peas - LOAD DATASET" , type={"csv"} )
if path is not None:
    path_df = pd.read_csv(path)
    st.write("DATASET")
    st.write(path_df)

    data=path_df.iloc[: , 0:7]

    #st.button("Click me for no reason")
#st.markdown(f'<h1 style="color:red;font-size:24px;">{"DATA CLEANING"}</h1>', unsafe_allow_html=True)
if(st.sidebar.button("HANDLING MISSING VALUES")):
    from sklearn.impute import SimpleImputer
    si = SimpleImputer(strategy="mean")
    path_df.iloc[: , 0:7]= si.fit_transform(path_df.iloc[: , 0:7])
    st.write("HANDLING MISSING VALUES")
    st.write(path_df)

if(st.sidebar.button("SCALING AND ENCODING")):
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    from sklearn.preprocessing import MinMaxScaler
    mms=MinMaxScaler()
    
   # path_df.iloc[: , 16:17]= le.fit_transform(path_df.iloc[: , 16:17])
    path_df.iloc[: , 0:7] = mms.fit_transform(path_df.iloc[: , 0:7])
    st.write("SCALING AND ENCODING")
    st.write(path_df)
    

#st.markdown(f'<h1 style="color:red;font-size:24px;">{"MODEL BUILDING"}</h1>', unsafe_allow_html=True)
if(st.sidebar.button("DECISION TREE MODEL")):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy")
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(path_df.iloc[: , 0:7] , path_df.iloc[: , 7:8], train_size=0.8)
    dtc.fit(x_train,y_train)
    pred = dtc.predict(x_test)
    from sklearn.metrics import confusion_matrix
    st.write("CONFUSION MATRIX")
    st.write(confusion_matrix(pred,y_test))
    fig, ax = plt.subplots(figsize=(10,4), dpi=100)
    classes = ['BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA']
    cm=confusion_matrix(pred,y_test)
    s=sns.heatmap(cm/np.sum(cm), annot=True,fmt='.2%', cmap='Blues')
    #fig, ax = plt.subplots()
    cmap = "tab20"
    linewidths=2
    sns.heatmap(data=cm,cmap=cmap,linewidths=linewidths)
    st.write(fig)
    
#st.markdown(f'<h1 style="color:red;font-size:24px;">{"MODEL ACCURACY"}</h1>', unsafe_allow_html=True)
if(st.sidebar.button("ACCURACY SCORE")):
    from sklearn.metrics import accuracy_score,f1_score
    st.write("ACCURACY SCORE OF DECISION TREE MODEL FOR DRY BEAN CLASSIFICATION")
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(path_df.iloc[: , 0:16] , path_df.iloc[: , 16:17], train_size=0.8)
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy")
    dtc.fit(x_train,y_train)
    pred = dtc.predict(x_test)
    ascore=accuracy_score(pred,y_test)
    st.write(ascore*100 , "%" )
    



        

