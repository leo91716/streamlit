
import subprocess
import numpy as np
import streamlit as st
from io import BytesIO
import pandas as pd
import os

from util.event import Event
from util.load_data import load_excel


filter_columns = [('BU', 1), ('ProdGroup', 2), ('MonthGrade', 2), ('QtyGrade', 2), ('NetUPriceGrade', 2)]
filtered_column = 'RepCust'
# filter_columns = ['c', 'd']
# filtered_column = 'b'

# session state key
__sales_data_file = 'sales_data_file'
__sales_data_sheet_names = 'sales_data_sheet_names'

__label_data_file = 'label_data_file'
__label_data_labels = 'label_data_labels'
__label_data_df = 'label_data_df'

__selected_sheet_name = 'selected_sheet_name'
__selected_labels = 'selected_labels'
__final_df = 'final_df'


def get_selected_df():
    '''
    Place file uploader at side bar and load selected excel file
    If file has more than one sheets, place selection box to selct loaded sheet
    Returns DataFrame if loads a sheet, otherwise None
    '''
    lang = st.session_state['lang_config']['pno']

    __load_sales_data_file(lang)
    __select_sheet(lang)
    __load_label_file(lang)
    __select_labels(lang)
    return __submit(lang)


def __load_sales_data_file(lang):
    st.sidebar.file_uploader(lang['upload_data_here'], type=['xlsx'], help=lang['upload_data_help'], on_change=__on_sales_data_changed, key=__sales_data_file)
    if not __is_key_set(__sales_data_file):
        st.warning(lang['upload_data_hint'])


def __on_sales_data_changed():
    if not __is_key_set(__sales_data_file):
        return

    file = st.session_state[__sales_data_file]
    st.session_state[__sales_data_sheet_names] = pd.ExcelFile(file).sheet_names


def __load_label_file(lang):
    st.sidebar.file_uploader(lang['upload_label_here'], type=['xlsx'], help=lang['upload_label_help'], on_change=__on_label_file_changed, key=__label_data_file)
    if not __is_key_set(__label_data_file):
        st.warning(lang['upload_label_hint'])


def __on_label_file_changed():
    if not __is_key_set(__label_data_file):
        return

    file = st.session_state[__label_data_file]
    df = load_excel(file)
    st.session_state[__label_data_df] = df

    d = {}
    for c, lim in filter_columns:
        a: np.ndarray = df[c].unique()
        a[[i is np.nan for i in a]] = 'nan'
        a.sort()
        d[c] = (a, lim)
    st.session_state[__label_data_labels] = d


def __select_sheet(lang):
    if not __is_key_set(__sales_data_sheet_names):
        return

    sheet_names = st.session_state[__sales_data_sheet_names]
    st.title(lang['select_bu'])
    st.session_state[__selected_sheet_name] = st.selectbox('', sheet_names)


def __select_labels(lang):
    if not __is_key_set(__label_data_labels):
        return
    st.title(lang['select_label'])
    st.session_state[__selected_labels] = {
        k: st.multiselect(k, v, v) if lim == 2 else [st.selectbox(k, v, 0)]
        for k, (v, lim) in st.session_state[__label_data_labels].items()
    }


def __submit(lang):
    if not __is_key_set(__sales_data_file) or not __is_key_set(__selected_sheet_name) or not __is_key_set(__selected_labels) or not __is_key_set(__label_data_file):
        return None, None, None
    
    predict_period = st.slider(lang['predict_period'], 1, 31, 31)

    updated = st.button(lang['submit'])
    if updated:
        with st.spinner(lang['parsing']):
            # get filtered from label file
            label_df = st.session_state[__label_data_df]
            sl = st.session_state[__selected_labels]
            for k, v in sl.items():
                label_df = label_df[label_df[k].isin(v)]

            label_df = label_df.sort_values('RepCust', ascending=False)
            label_df = label_df.drop_duplicates(subset='RepCust', keep='first')
            filtered_company_names = label_df[filtered_column]

            # filter sales data
            df = load_excel(st.session_state[__sales_data_file], st.session_state[__selected_sheet_name])
            df = df[df[filtered_column].isin(filtered_company_names)]

        st.session_state[__final_df] = df

    if __final_df not in st.session_state:
        st.session_state[__final_df] = None

    return st.session_state[__final_df], updated, predict_period


def __is_key_set(key):
    return key in st.session_state and st.session_state[key] is not None
