import streamlit as st
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import math
st.markdown(f'<h1 style="color:BLUE;font-size:45px;">{"COTTON SEEDS - CLASSIFICATION"}</h1>', unsafe_allow_html=True)
#st.title("Black-Eyed Peas - DRY BEAN CLASSIFICATION")
import streamlit as st
st.image("CS.jpg")
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
path = st.sidebar.file_uploader("COTTON SEEDS - LOAD DATASET" , type={"csv"} )
if path is not None:
    path_df = pd.read_csv(path)
    st.markdown(f'<h1 style="color:BLUE;font-size:17px;">{"COTTON SEED DATASET"}</h1>', unsafe_allow_html=True)
    #st.write("DATASET")
    st.write(path_df)

    data=path_df.iloc[: , 0:7]
    #st.button("Click me for no reason")
#st.markdown(f'<h1 style="color:red;font-size:24px;">{"DATA CLEANING"}</h1>', unsafe_allow_html=True)
if(st.sidebar.button("HANDLING MISSING VALUES")):
    from sklearn.impute import SimpleImputer
    si = SimpleImputer(strategy="mean")
    path_df.iloc[: , 0:7]= si.fit_transform(path_df.iloc[: , 0:7])
    #st.write("HANDLING MISSING VALUES")
    st.markdown(f'<h1 style="color:BLUE;font-size:20px;">{"HANDLING MISSING VAUES "}</h1>', unsafe_allow_html=True)
    st.image("MARKS.jpg")
    st.write(path_df)
if(st.sidebar.button("SCALING AND ENCODING")):
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    from sklearn.preprocessing import MinMaxScaler
    mms=MinMaxScaler()
   # path_df.iloc[: , 16:17]= le.fit_transform(path_df.iloc[: , 16:17])
    path_df.iloc[: , 0:7] = mms.fit_transform(path_df.iloc[: , 0:7])
    st.markdown(f'<h1 style="color:BLUE;font-size:20px;">{"SCALING & ENCODING"}</h1>', unsafe_allow_html=True)
    st.image("123.png")
    st.write(path_df)
#st.markdown(f'<h1 style="color:red;font-size:24px;">{"MODEL BUILDING"}</h1>', unsafe_allow_html=True)
if(st.sidebar.button("NTGT MODEL")):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy")
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(path_df.iloc[: , 0:7] , path_df.iloc[: , 7:8], train_size=0.8)
    dtc.fit(x_train,y_train)
    pred = dtc.predict(x_test)
    from sklearn.metrics import confusion_matrix
    #st.write("CONFUSION MATRIX")
    st.markdown(f'<h1 style="color:BLUE;font-size:20px;">{"CONFUSION MATRIX"}</h1>', unsafe_allow_html=True)
    st.write(confusion_matrix(pred,y_test))
    fig, ax = plt.subplots(figsize=(10,4), dpi=100)
    cm=confusion_matrix(pred,y_test)
    s=sns.heatmap(cm/np.sum(cm), annot=True,fmt='.2%', cmap='Blues')
    #fig, ax = plt.subplots()
    cmap = "tab20"
    linewidths=2
    sns.heatmap(data=cm,cmap=cmap,linewidths=linewidths)
    st.write(fig)
#st.markdown(f'<h1 style="color:red;font-size:24px;">{"MODEL ACCURACY"}</h1>', unsafe_allow_html=True)
if(st.sidebar.button("NTGT MODEL ACCURACY SCORE")):
    from sklearn.metrics import accuracy_score,f1_score
    st.markdown(f'<h1 style="color:BLUE;font-size:17px;">{"ACCURACY SCORE OF NOVEL TANGENT - DECISION TREE MODEL FOR COTTON SEED CLASSIFICATION"}</h1>', unsafe_allow_html=True)
    #st.write("ACCURACY SCORE OF DECISION TREE MODEL FOR COTTON SEED CLASSIFICATION")
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(path_df.iloc[: , 0:7] , path_df.iloc[: , 7:8], train_size=0.8)
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy")
    dtc.fit(x_train,y_train)
    pred = dtc.predict(x_test)
    st.ascore1=accuracy_score(pred,y_test)
    st.write(st.ascore1*100 , "%" )
    
#st.markdown(f'<h1 style="color:red;font-size:24px;">{"MODEL BUILDING"}</h1>', unsafe_allow_html=True)
if(st.sidebar.button("RKN MODEL")):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(path_df.iloc[: , 0:7] , path_df.iloc[: , 7:8], train_size=0.8)
    neigh.fit(x_train,y_train)
    pred = neigh.predict(x_test)
    from sklearn.metrics import confusion_matrix
    #st.write("CONFUSION MATRIX")
    st.markdown(f'<h1 style="color:BLUE;font-size:20px;">{"CONFUSION MATRIX"}</h1>', unsafe_allow_html=True)
    st.write(confusion_matrix(pred,y_test))
    fig, ax = plt.subplots(figsize=(10,4), dpi=100)
    cm=confusion_matrix(pred,y_test)
    s=sns.heatmap(cm/np.sum(cm), annot=True,fmt='.2%', cmap='Blues')
    #fig, ax = plt.subplots()
    cmap = "tab20"
    linewidths=2
    sns.heatmap(data=cm,cmap=cmap,linewidths=linewidths)
    st.write(fig)

if(st.sidebar.button("RKN MODEL ACCURACY SCORE")):
    from sklearn.metrics import accuracy_score,f1_score
    st.markdown(f'<h1 style="color:BLUE;font-size:20px;">{"ACCURACY SCORE OF RANDOM - K-VALUE KNN MODEL FOR COTTON SEED CLASSIFICATION"}</h1>', unsafe_allow_html=True)
    #st.write("ACCURACY SCORE OF KNN MODEL FOR COTTON SEED CLASSIFICATION")
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    x_train,x_test,y_train,y_test = train_test_split(path_df.iloc[: , 0:7] , path_df.iloc[: , 7:8], train_size=0.8)
    from sklearn.tree import DecisionTreeClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train,y_train)
    pred = neigh.predict(x_test)
    st.ascore2=accuracy_score(pred,y_test)
    st.write(st.ascore2*100 , "%" )




