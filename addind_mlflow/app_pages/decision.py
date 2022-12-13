# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:27:59 2020

@author: NBGhoshSu3
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from IPython.display import Image
#from sklearn.externals.six import StringIO
import pydotplus, graphviz
import six
import sys
sys.modules['sklearn.externals.six'] = six
from six import StringIO

def app():
    #@st.cache(suppress_st_warning=True)
    max_depth=3
    max_leaf_nodes = 100
    min_samples_split = 5
    min_samples_leaf =  5
    criterion = 'gini'
	
    def highlight_max(data, color='yellow'):
        attr = 'background-color: {}'.format(color)
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            is_max = data == data.max().max()
            return pd.DataFrame(np.where(is_max, attr, ''),index=data.index, columns=data.columns)


    st.title('MiAutoML決策樹 - Hyper Parameter Tuning')
    st.write('### 所選資料表')
	
	# asking for file
    file_upload = st.sidebar.file_uploader('Upload your Data Her', type=['csv'], help="Only `csv` Please")
    name = st.sidebar.selectbox('Selcect Sample Data', options=['titanic1','heart_v2','None'], help='Select Data From Here')

    # smple file getting function
    def get_dataset(name, sample=True, custome=False ):
        #max_depth =3
        try:
            if sample:
                if name=='heart_v2': # matchin user choose file 
                    df = pd.read_csv('heart_v2.csv')
                elif name=='titanic1': # matchin user choose file 
                    df = pd.read_csv('titanic1.csv')
                    return  df # retruning the data frame

            if custome:
                df = pd.read_csv(file_upload)
                return df
        except Exception:
            print('error n load_dataset')

    # giving result by choosing dataset, custome or sample
    if file_upload is None:
        df = get_dataset(name=name) 
    else:
        df = get_dataset(name=name,custome=True, sample=False)

    if df is not None:
        st.write('Your Uploaded Data')        
        #st.dataframe(df)
        
        st.dataframe(df.style.apply(highlight_max,subset=['Survived']))

    else:
        st.warning('1.請用左側上傳功能上傳一個檔案')

    try:
        st.write('-'*100)
        X = df.drop('Survived', axis=1)
        y = df['Survived'].copy()
    except:
        st.warning('2.或選擇一組範例資料')
		

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    #else:
    #    st.warning('2.或選擇一組範例資料')

    #df = pd.read_csv('./heart_v2.csv')
    #st.dataframe(df.style.apply(highlight_max,subset=['heart disease']))
    #st.write('-'*100)
    #X = df.drop('heart disease', axis=1)
    #y = df['heart disease'].copy()

    # Test Train Split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    #max_depth = st.sidebar.slider('Maximum Depth', min_value=1, max_value=25, step=1, value=3)
    #max_leaf_nodes = st.sidebar.slider('Maximum Leaves', min_value=2, max_value=100, step=1, value=100)
    #min_samples_split = st.sidebar.slider('Minimum Samples Before Split', min_value=2, max_value=200, step=1, value=5)
    #min_samples_leaf = st.sidebar.slider('Min Samples In Each Leaf', min_value=1, max_value=200, step=1, value=5)
    #criterion = st.sidebar.selectbox('Spliting Criterion', ['gini', 'entropy'])


    def classify(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion):
        dt = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)
        return dt.fit(X_train, y_train)

    #@st.cache(suppress_st_warning=True)
    def get_dt_graph(dt_classifier):
        dot_data = StringIO()
        export_graphviz(dt_classifier, out_file=dot_data, filled=True, rounded=True,feature_names=X.columns, class_names=['survived', 'dead'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        return graph

    #@st.cache(suppress_st_warning=True)
    def evaluate_model(dt_classifier):
        y_train_pred = dt_classifier.predict(X_train)
        y_test_pred = dt_classifier.predict(X_test)
        st.write('### Training Dataset Performance')
        st.write('Accuracy : ', 100 * np.round(accuracy_score(y_train, y_train_pred), 3))
        st.write('#### Confusion Matrix')
        confusion = confusion_matrix(y_train, y_train_pred)
        st.write(confusion)
        TP = confusion[1, 1]  # true positive
        TN = confusion[0, 0]  # true negatives
        FP = confusion[0, 1]  # false positives
        FN = confusion[1, 0]  # false negatives

        sensitivity = TP/(FN + TP)
        specificity = TN/(FP + TN)
        falsePositiveRate = FP/(FP + TN)
        positivePredictivePower = TP/(TP + FP)
        negativePredictivePower = TN/(TN + FN)
        st.write('Sensitivity / Recall: ', round(100*sensitivity, 3), '%')
        st.write('Specificity : ',  round(100*specificity, 3), '%')
        st.write('False Positive Rate : ',  round(100*falsePositiveRate, 3), '%')
        st.write('Precision / Positive Predictive Power : ',round(100*positivePredictivePower, 3), '%')
        st.write('Negative Predictive Power : ',  round(100*negativePredictivePower, 3), '%')

        st.write("-"*60)
        st.write('### Test Set Performance')
        st.write('Accuracy : ', 100*np.round(accuracy_score(y_test, y_test_pred), 3))
        st.write('#### Confusion Matrix')
        confusion = confusion_matrix(y_test, y_test_pred)
        st.write(confusion)

        TP = confusion[1, 1]  # true positive
        TN = confusion[0, 0]  # true negatives
        FP = confusion[0, 1]  # false positives
        FN = confusion[1, 0]  # false negatives

        sensitivity = TP/(FN + TP)
        specificity = TN/(FP + TN)
        falsePositiveRate = FP/(FP + TN)
        positivePredictivePower = TP/(TP + FP)
        negativePredictivePower = TN/(TN + FN)
        st.write('Sensitivity / Recall: ', round(100*sensitivity, 3), '%')
        st.write('Specificity : ',  round(100*specificity, 3), '%')
        st.write('False Positive Rate : ',  round(100*falsePositiveRate, 3), '%')
        st.write('Precision / Positive Predictive Power : ', round(100*positivePredictivePower, 3), '%')
        st.write('Negative Predictive Power : ',  round(100*negativePredictivePower, 3), '%')
    
        st.write("-"*100)
		


    max_depth = st.sidebar.slider('Maximum Depth', min_value=1, max_value=25, step=1, value=3) 
    max_leaf_nodes = st.sidebar.slider('Maximum Leaves', min_value=2, max_value=100, step=1, value=100)
    min_samples_split = st.sidebar.slider('Minimum Samples Before Split', min_value=2, max_value=200, step=1, value=5)
    min_samples_leaf = st.sidebar.slider('Min Samples In Each Leaf', min_value=1, max_value=200, step=1, value=5)
    criterion = st.sidebar.selectbox('Spliting Criterion', ['gini', 'entropy'])

    dt = classify(max_depth, max_leaf_nodes, min_samples_split, min_samples_leaf, criterion)

    graph = get_dt_graph(dt)
    st.write('### 決策樹結果')
    st.image(graph.create_png(), width=1000)
    st.write("-"*60)

    evaluate_model(dt)


















