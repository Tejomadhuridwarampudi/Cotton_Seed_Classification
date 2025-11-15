import streamlit as st
import numpy as np
import pandas as pd

# ML imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.markdown(
    '<h1 style="color:BLUE;font-size:45px;">COTTON SEEDS - CLASSIFICATION</h1>',
    unsafe_allow_html=True
)

# Image
st.image("CS.jpg")

# File Upload
path = st.sidebar.file_uploader("COTTON SEEDS - LOAD DATASET", type={"csv"})

if path:
    path_df = pd.read_csv(path)

    st.markdown(
        '<h1 style="color:BLUE;font-size:20px;">COTTON SEED DATASET</h1>',
        unsafe_allow_html=True
    )
    st.write(path_df)

# -------------------------
# HANDLE MISSING VALUES
# -------------------------
if st.sidebar.button("HANDLING MISSING VALUES"):
    si = SimpleImputer(strategy="mean")
    path_df.iloc[:, 0:16] = si.fit_transform(path_df.iloc[:, 0:16])

    st.markdown(
        '<h1 style="color:BLUE;font-size:20px;">HANDLING MISSING VALUES</h1>',
        unsafe_allow_html=True
    )
    st.image("MARKS.jpg")
    st.write(path_df)

# -------------------------
# SCALING & ENCODING
# -------------------------
if st.sidebar.button("SCALING AND ENCODING"):
    mms = MinMaxScaler()
    path_df.iloc[:, 0:16] = mms.fit_transform(path_df.iloc[:, 0:16])

    st.markdown(
        '<h1 style="color:BLUE;font-size:20px;">SCALING & ENCODING</h1>',
        unsafe_allow_html=True
    )
    st.image("123.png")
    st.write(path_df)

# -------------------------
# NTGT MODEL (Decision Tree)
# -------------------------
if st.sidebar.button("NTGT MODEL"):
    x_train, x_test, y_train, y_test = train_test_split(
        path_df.iloc[:, 0:16], path_df.iloc[:, 16:], train_size=0.8
    )

    dtc = DecisionTreeClassifier(criterion="entropy")
    dtc.fit(x_train, y_train)
    pred = dtc.predict(x_test)

    cm = confusion_matrix(y_test, pred)

    st.markdown(
        '<h1 style="color:BLUE;font-size:20px;">CONFUSION MATRIX</h1>',
        unsafe_allow_html=True
    )
    st.write(cm)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
    st.pyplot(fig)

# Accuracy of NTGT
if st.sidebar.button("NTGT MODEL ACCURACY SCORE"):
    x_train, x_test, y_train, y_test = train_test_split(
        path_df.iloc[:, 0:16], path_df.iloc[:, 16:], train_size=0.8
    )

    dtc = DecisionTreeClassifier(criterion="entropy")
    dtc.fit(x_train, y_train)
    pred = dtc.predict(x_test)

    acc = accuracy_score(y_test, pred)

    st.markdown(
        '<h1 style="color:BLUE;font-size:20px;">ACCURACY SCORE OF NTGT MODEL</h1>',
        unsafe_allow_html=True
    )
    st.write(f"{acc * 100:.2f}%")

# -------------------------
# RKN MODEL (KNN)
# -------------------------
if st.sidebar.button("RKN MODEL"):
    x_train, x_test, y_train, y_test = train_test_split(
        path_df.iloc[:, 0:16], path_df.iloc[:, 16:], train_size=0.8
    )

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    pred = neigh.predict(x_test)

    cm = confusion_matrix(y_test, pred)

    st.markdown(
        '<h1 style="color:BLUE;font-size:20px;">CONFUSION MATRIX</h1>',
        unsafe_allow_html=True
    )
    st.write(cm)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
    st.pyplot(fig)

# Accuracy of RKN
if st.sidebar.button("RKN MODEL ACCURACY SCORE"):
    x_train, x_test, y_train, y_test = train_test_split(
        path_df.iloc[:, 0:16], path_df.iloc[:, 16:], train_size=0.8
    )

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    pred = neigh.predict(x_test)

    acc = accuracy_score(y_test, pred)

    st.markdown(
        '<h1 style="color:BLUE;font-size:20px;">ACCURACY SCORE OF RKN MODEL</h1>',
        unsafe_allow_html=True
    )
    st.write(f"{acc * 100:.2f}%")
