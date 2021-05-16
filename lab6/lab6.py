#!/usr/bin/env python
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error


st.header('Датасет')
main_status = st.text('')
read_state = st.text('Чтение датасета...')
data = pd.read_csv("data/Heart_Disease_Data.csv", sep=',')
data2 = data.copy()
read_state.text('Датасет загружен!')

st.subheader('head:')
st.write(data.head())

if st.checkbox('Обработать пропуски'):
    empty_processing_state = st.text('Обработка пропусков..')
    data2 = data2.drop(data2[data2.ca == '?'].index)
    data2 = data2.drop(data2[data2.thal == '?'].index)
    data2['ca'] = data2['ca'].astype(int)
    data2['thal'] = data2['thal'].astype(int)
    empty_processing_state.text('Пропуски обработаны!')


test_size = st.sidebar.slider("test_size", 0.1, 0.9, value = 0.3)
n_estimators = st.sidebar.slider("n_estimators", 1, 20, value=5)
random_state = st.sidebar.slider("random_state", 1, 20, value=10)

target_option = st.sidebar.selectbox('Target:', data2.columns)
feature_cols = []
st.sidebar.subheader('Features:')
for col in data2.columns:
    cb = st.sidebar.checkbox(col, value=True)
    if cb:
        feature_cols.append(col)

st.sidebar.subheader('Мастабирование')
if st.sidebar.checkbox('MinMaxScaler'):
    sc2 = MinMaxScaler()
    if target_option != 'age':
        data2['age'] = sc2.fit_transform(data2[['age']])
    if target_option != 'trestbps':
        data2['trestbps'] = sc2.fit_transform(data2[['trestbps']])
    if target_option != 'chol':
        data2['chol'] = sc2.fit_transform(data2[['chol']])
    if target_option != 'thalach':
        data2['thalach'] = sc2.fit_transform(data2[['thalach']])
    if target_option != 'oldpeak':
        data2['oldpeak'] = sc2.fit_transform(data2[['oldpeak']])
    st.subheader('Мастабирование:')
    st.write(data2.head())


main_status.text('В процессе обучения...')
data_X = data2.loc[:,feature_cols]
data_Y = data2.loc[:, target_option]
data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(
    data_X, data_Y,test_size=test_size, random_state=1)

bc = BaggingClassifier(n_estimators=n_estimators, oob_score=True, random_state=random_state)
bc.fit(data_X_train, data_y_train)

rfc = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=random_state)
rfc.fit(data_X_train, data_y_train)

abc = AdaBoostClassifier(n_estimators=n_estimators, algorithm='SAMME', random_state=random_state)
abc.fit(data_X_train, data_y_train)

gbc = GradientBoostingClassifier(random_state=random_state)
gbc.fit(data_X_train, data_y_train)

main_status.text('Обучено!')
metrics = [mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error]
metr = [i.__name__ for i in metrics]
metrics_ms = st.sidebar.multiselect("Метрики", metr)

methods = [bc, rfc, abc, gbc]
md = [i.__class__.__name__ for i in methods]
methods_ms = st.sidebar.multiselect("Методы обучения", md)

selMethods = []
for i in methods_ms:
    for j in methods:
        if i == j.__class__.__name__:
            selMethods.append(j)

selMetrics = []
for i in metrics_ms:
    for j in metrics:
        if i == j.__name__:
            selMetrics.append(j)

st.header('Оценка')
for name in selMetrics:
    st.subheader(name.__name__)
    for func in selMethods:
        y_pred = func.predict(data_X_test)
        st.text("{} - {}".format(func.__class__.__name__, name(y_pred, data_y_test)))

