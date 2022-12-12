# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:27:59 2020

@author: NBGhoshSu3
"""

from app_pages.app_page import AppPage
from six import StringIO
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
import pydotplus
import graphviz
import six
import sys
sys.modules['sklearn.externals.six'] = six


class Decision1Page(AppPage):
    @staticmethod
    def _run_page():
        app()

    @staticmethod
    def get_name():
        return st.session_state['lang_config']['decision1']['name']


def app():
    lang = st.session_state['lang_config']['decision1']

    # @st.cache(suppress_st_warning=True)
    max_depth = 5
    max_leaf_nodes = 100
    min_samples_split = 5
    min_samples_leaf = 5
    criterion = 'gini'

    def highlight_max(data, color='yellow'):
        attr = 'background-color: {}'.format(color)
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.min()
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            is_max = data == data.min().min()
            return pd.DataFrame(np.where(is_max, attr, ''), index=data.index, columns=data.columns)

    st.title(lang['title'])
    st.write(lang['selected_data'])

    # asking for file
    file_upload = st.sidebar.file_uploader(lang['upload_here'], type=['csv'], help=lang['upload_help'])
    name = 'c3'

    # smple file getting function

    df = pd.read_csv('data\c3.csv')

    # giving result by choosing dataset, custome or sample

    st.dataframe(df.style.apply(highlight_max, subset=['MonthGrade']))

    st.write('-' * 100)
    X = df.drop(['MonthGrade', 'BU', 'RepCust'], axis=1)
    y = df['MonthGrade'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    # else:
    #    st.warning('2.或選擇一組範例資料')

    #df = pd.read_csv('./heart_v2.csv')
    #st.dataframe(df.style.apply(highlight_max,subset=['heart disease']))
    # st.write('-'*100)
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

    # @st.cache(suppress_st_warning=True)
    def get_dt_graph(dt_classifier):
        dot_data = StringIO()
        export_graphviz(dt_classifier, out_file=dot_data, filled=True, rounded=True, feature_names=X.columns, class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        return graph

    # @st.cache(suppress_st_warning=True)
    def evaluate_model(dt_classifier):
        y_train_pred = dt_classifier.predict(X_train)
        y_test_pred = dt_classifier.predict(X_test)
        st.write(lang['train_performance'])
        st.write(lang['train_accuracy'], 100 * np.round(accuracy_score(y_train, y_train_pred), 3))
        st.write(lang['confusion_matrix'])
        confusion = confusion_matrix(y_train, y_train_pred)
        st.write(confusion)
        TP = confusion[1, 1]  # true positive
        TN = confusion[0, 0]  # true negatives
        FP = confusion[0, 1]  # false positives
        FN = confusion[1, 0]  # false negatives

        sensitivity = TP / (FN + TP)
        specificity = TN / (FP + TN)
        falsePositiveRate = FP / (FP + TN)
        positivePredictivePower = TP / (TP + FP)
        negativePredictivePower = TN / (TN + FN)
        st.write(lang['sensitivity'], round(100 * sensitivity, 3), '%')
        st.write(lang['specificity'], round(100 * specificity, 3), '%')
        st.write(lang['false_positive_rate'], round(100 * falsePositiveRate, 3), '%')
        st.write(lang['precision'], round(100 * positivePredictivePower, 3), '%')
        st.write(lang['negative_predictive_power'], round(100 * negativePredictivePower, 3), '%')

        st.write("-"*60)
        st.write(lang['test_performance'])
        st.write(lang['train_accuracy'], 100 * np.round(accuracy_score(y_test, y_test_pred), 3))
        st.write(lang['confusion_matrix'])
        confusion = confusion_matrix(y_test, y_test_pred)
        st.write(confusion)

        TP = confusion[1, 1]  # true positive
        TN = confusion[0, 0]  # true negatives
        FP = confusion[0, 1]  # false positives
        FN = confusion[1, 0]  # false negatives

        sensitivity = TP / (FN + TP)
        specificity = TN / (FP + TN)
        falsePositiveRate = FP / (FP + TN)
        positivePredictivePower = TP / (TP + FP)
        negativePredictivePower = TN / (TN + FN)
        st.write(lang['sensitivity'], round(100 * sensitivity, 3), '%')
        st.write(lang['specificity'], round(100 * specificity, 3), '%')
        st.write(lang['false_positive_rate'], round(100 * falsePositiveRate, 3), '%')
        st.write(lang['precision'], round(100 * positivePredictivePower, 3), '%')
        st.write(lang['negative_predictive_power'], round(100 * negativePredictivePower, 3), '%')

        st.write("-"*100)

    max_depth = st.sidebar.slider(lang['max_depth'], min_value=1, max_value=25, step=1, value=5)
    max_leaf_nodes = st.sidebar.slider(lang['max_leaves'], min_value=2, max_value=100, step=1, value=100)
    min_samples_split = st.sidebar.slider(lang['min_samples_bf_split'], min_value=2, max_value=200, step=1, value=5)
    min_samples_leaf = st.sidebar.slider(lang['min_samples_in_leaf'], min_value=1, max_value=200, step=1, value=5)
    criterion = st.sidebar.selectbox(lang['split_criterion'], ['gini', 'entropy'])

    dt = classify(max_depth, max_leaf_nodes, min_samples_split, min_samples_leaf, criterion)

    graph = get_dt_graph(dt)
    st.write(lang['decision_result'])
    st.image(graph.create_png(), width=1000)
    st.write("-" * 60)

    evaluate_model(dt)
