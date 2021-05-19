#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from IPython.display import Image

import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score 
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC, NuSVC, LinearSVC, OneClassSVM, SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

class MetricLogger:
    
    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
            'alg': pd.Series([], dtype='str'),
            'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric']==metric)&(self.df['alg']==alg)].index, inplace = True)
        # Добавление нового значения
        temp = [{'metric':metric, 'alg':alg, 'value':value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric']==metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values
    
    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5, 
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a,b in zip(pos, array_metric):
            plt.text(0.5, a-0.05, str(round(b,3)), color='white')
        return fig 

def regr_train_model(model_name, model, regrMetricLogger):
    model.fit(data_X_train, data_y_train)
    y_pred = model.predict(data_X_test)
    
    mae = mean_absolute_error(data_y_test, y_pred)
    mse = mean_squared_error(data_y_test, y_pred)
    r2 = r2_score(data_y_test, y_pred)

    regrMetricLogger.add('MAE', model_name, mae)
    regrMetricLogger.add('MSE', model_name, mse)
    regrMetricLogger.add('R2', model_name, r2)    
    
    return '{} \t MAE={}, MSE={}, R2={}'.format(
        model_name, round(mae, 3), round(mse, 3), round(r2, 3))

@st.cache
def load_data():
    read_data = pd.read_csv("data/kc_house_data.csv", sep=',')
    read_data.drop(['id', 'date'], axis=1, inplace=True)
    return read_data


st.header('Датасет')
read_state = st.text('Чтение датасета...')
data = load_data().copy()
read_state.text('Датасет загружен!')
st.subheader('head:')
st.write(data.head())

regr_models = []


if st.sidebar.checkbox('Limit rows(for better performance)'):
    max_rows = st.sidebar.slider('Max rows', 200, len(data), 2000)
    data = data.head(max_rows)

if st.sidebar.checkbox('MinMaxScaler'):
    scale_cols = ['sqft_living', 'sqft_lot', 'sqft_above',
            'sqft_basement', 'lat', 'long',
            'sqft_living15', 'sqft_lot15', 'bedrooms',
            'bathrooms', 'view', 'grade', 'floors', 'yr_renovated', 'yr_built', 'condition'
            ]
    sc = MinMaxScaler()
    sc_data = sc.fit_transform(data[scale_cols])
    # Добавим масштабированные данные в набор данных
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        data[col] = sc_data[:,i]
    st.subheader('Масштабированный датасет:')
    st.write(data.head())

test_size = st.sidebar.slider('test_size', 0.1, 0.9, value = 0.3)

feature_cols = [
    'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view',
    'grade', 'yr_renovated', 'sqft_living', 'sqft_above',
    'sqft_basement', 'lat', 'sqft_living15'
]
data_X = data.loc[:,feature_cols]
data_Y = data.loc[:, 'price']
data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X, data_Y,test_size=test_size, random_state=1)

# Сохранение метрик
regrMetricLogger = MetricLogger()

if st.sidebar.checkbox('LinearRegression'):
    regr_train_model('LR', LinearRegression(), regrMetricLogger)

if st.sidebar.checkbox('KNeighborsRegression'):
    n_neighbors = 5
    n_neighbors = st.sidebar.slider("n_neighbors", 1, 100, value = 5)
    regr_train_model('KNN', KNeighborsRegressor(n_neighbors=n_neighbors), regrMetricLogger)
    if st.sidebar.checkbox("KNeighborsRegression(Best)"):
        n_neighbors_search_1 = st.sidebar.slider("n starts from", 1, 100)
        n_neighbors_search_2 = st.sidebar.slider("n ends with", 1, 100, value=10)
        n_neighbors_search_step = st.sidebar.slider("n step", 1, 100, 1)
        n_range = np.array(range(n_neighbors_search_1, n_neighbors_search_2, n_neighbors_search_step))
        kn_tuned_parameters = [{'n_neighbors': n_range}]

        gs_kn = GridSearchCV(KNeighborsRegressor(), kn_tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        gs_kn.fit(data_X_train, data_y_train)
        regr_train_model('KNN(Best)', gs_kn.best_estimator_, regrMetricLogger)
        st.subheader('Best model for KNeighborsRegressor:')
        st.write(gs_kn.best_estimator_)
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(n_range, gs_kn.cv_results_['mean_test_score'])
        st.pyplot(fig)
    st.sidebar.markdown("""---""")


if st.sidebar.checkbox('LinearSVR'):
    regr_train_model('LinearSVR', LinearSVR(), regrMetricLogger)
    if st.sidebar.checkbox('LinearSVR(Best)'):
        c_start = st.sidebar.slider("c starts from", 1, 100000, value=1)
        c_ends = st.sidebar.slider("c ends with", 1, 100000, value=10000)
        c_step = st.sidebar.slider("c step", 1, 1000, value=1000)
        lsvr_c_range = np.array(range(c_start, c_ends, c_step))
        lsvr_tuned_params = [{'C': lsvr_c_range}]

        gs_lsvr = GridSearchCV(LinearSVR(), lsvr_tuned_params, cv=5, scoring='neg_mean_squared_error')
        gs_lsvr.fit(data_X_train, data_y_train)
        regr_train_model('LinearSVR(Best)', gs_lsvr.best_estimator_, regrMetricLogger)
        st.subheader('Best model for LinearSVR:')
        st.write(gs_lsvr.best_estimator_)
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(lsvr_c_range, gs_lsvr.cv_results_['mean_test_score'])
        st.pyplot(fig)
    st.sidebar.markdown("""---""")


if st.sidebar.checkbox('DecisionTreeRegressor'):
    regr_train_model('Tree', DecisionTreeRegressor(), regrMetricLogger)

if st.sidebar.checkbox('RandomForestRegressor'):
    regr_train_model('RF', RandomForestRegressor(), regrMetricLogger)
    if st.sidebar.checkbox('RandomForestRegressor(Best)'):
        n_estimators_search_1 = st.sidebar.slider("ne starts from", 100, 1000, value=100)
        n_estimators_search_2 = st.sidebar.slider("ne ends with", 1, 1000, value=101)
        n_estimators_search_step = st.sidebar.slider("ne step", 100, 1000, value=100)
        n_estimators_range = np.array(range(n_estimators_search_1, n_estimators_search_2, n_estimators_search_step))
        rf_tuned_parameters = [{'n_estimators': n_estimators_range}]
        
        gs_rf = GridSearchCV(RandomForestRegressor(), rf_tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        gs_rf.fit(data_X_train, data_y_train)
        regr_train_model('RF(Best)', gs_rf.best_estimator_, regrMetricLogger)
        st.subheader('Best model for RandomForestRegressor:')
        st.write(gs_rf.best_estimator_)
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(n_estimators_range, gs_rf.cv_results_['mean_test_score'])
        st.pyplot(fig)
    st.sidebar.markdown("""---""")



if st.sidebar.checkbox('GradientBoostingRegressor'):
    regr_train_model('GB', GradientBoostingRegressor(), regrMetricLogger)
    if st.sidebar.checkbox('GradientBoostingRegressor(Best)'):
        n_estimators_search_1 = st.sidebar.slider("nee starts from", 100, 1000, value=100)
        n_estimators_search_2 = st.sidebar.slider("nee ends with", 1, 1000, value=101)
        n_estimators_search_step = st.sidebar.slider("nee step", 100, 1000, value=100)
        n_estimators_range = np.array(range(n_estimators_search_1, n_estimators_search_2, n_estimators_search_step))
        gb_tuned_parameters = [{'n_estimators': n_estimators_range}]
        
        gs_gb = GridSearchCV(GradientBoostingRegressor(), gb_tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        gs_gb.fit(data_X_train, data_y_train)
        regr_train_model('GB(Best)', gs_gb.best_estimator_, regrMetricLogger)
        st.subheader('Best model for GradientBoostingRegressor:')
        st.write(gs_gb.best_estimator_)
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(n_estimators_range, gs_gb.cv_results_['mean_test_score'])
        st.pyplot(fig)
    st.sidebar.markdown("""---""")

st.sidebar.header('Metrics')
if st.sidebar.checkbox('MAE'):
    fig = regrMetricLogger.plot('Метрика: ' + 'MAE', 'MAE', ascending=False, figsize=(7, 6))
    st.pyplot(fig)

if st.sidebar.checkbox('MSE'):
    fig = regrMetricLogger.plot('Метрика: ' + 'MSE', 'MSE', ascending=False, figsize=(7, 6))
    st.pyplot(fig)

if st.sidebar.checkbox('r2'):
    fig = regrMetricLogger.plot('Метрика: ' + 'R2', 'R2', ascending=True, figsize=(7, 6))
    st.pyplot(fig)
