
import subprocess
import numpy as np
import streamlit as st
from io import BytesIO
import pandas as pd
import os

from util.event import Event


filter_columns = [('BU', 1), ('ProdGroup', 2), ('MonthGrade', 2), ('QtyGrade', 2), ('NetUPriceGrade', 2)]
filtered_column = 'RepCust'
# filter_columns = ['c', 'd']
# filtered_column = 'b'

# session state key
__sales_data_file = 'sales_data_upload'
__sales_data_df = 'sales_data_df'
__sales_data_excelfile = 'sales_data_excelfile'
__sales_data_sheet_names = 'sheet_names'
__sales_data_selected_sheet = 'sales_data_select_sheet'
__sales_data_file = 'sales_data_file'


__label_data_upload = 'label_data_upload'
__label_data_df = 'label_data_df'

__unique_labels = 'unique_labels'
__selected_labels = 'selected_labels'

__filtered_company_names = 'filtered_company_names'

__filtered_df = 'filtered_df'
__df_updated = 'df_updated'

# event
sales_data_sheet_update_event = Event()

filtered_company_names_event = Event()


def get_selected_df():
    '''
    Place file uploader at side bar and load selected excel file
    If file has more than one sheets, place selection box to selct loaded sheet
    Returns DataFrame if loads a sheet, otherwise None
    '''

    st.sidebar.file_uploader('Upload Customer Labels Here', type=['xlsx'], help="Only `xlsx` Please", on_change=__on_label_upload_change, key=__label_data_upload)
    if not __is_key_set(__label_data_upload):
        st.warning('Please upload customer label')

    if __is_key_set(__unique_labels):
        st.write("Please select filter")
        st.session_state[__selected_labels] = {
            k: st.multiselect(k, v, v) if lim == 2 else [st.selectbox(k, v, 0)]
            for k, (v, lim) in st.session_state[__unique_labels].items()
        }

    st.sidebar.file_uploader('Upload Sales Data Here', type=['xlsx'], help="Only `xlsx` Please", on_change=__on_sales_upload_change, key=__sales_data_file)
    if not __is_key_set(__sales_data_file):
        st.warning('Please upload sales data')

    __select_sheet()

    st.session_state[__df_updated] = False
    if __is_key_set(__selected_labels) and __is_key_set(__label_data_df) and __is_key_set(__sales_data_df):
        if st.button('Submit'):

            df, updated = __submit_df()
            return df, updated

    return None, st.session_state[__df_updated]

def __submit_df():
    label_df = st.session_state[__label_data_df]
    sl = st.session_state[__selected_labels]
    for k, v in sl.items():
        label_df = label_df[label_df[k].isin(v)]

    label_df = label_df.sort_values('RepCust', ascending=False)
    label_df = label_df.drop_duplicates(subset='RepCust', keep='first')
    st.session_state[__filtered_company_names] = label_df[filtered_column]
    filtered_company_names_event.invoke()
    st.session_state[__df_updated] = True

def __on_sales_upload_change():
    if not __is_key_set(__sales_data_file):
        st.session_state[__sales_data_sheet_names] = None
    else:
        datafile = st.session_state[__sales_data_file]
        st.session_state[__sales_data_sheet_names] = __get_excelfile(datafile).sheet_names


def __on_label_upload_change():
    labelfile = st.session_state[__label_data_upload]
    if labelfile is None:
        st.session_state[__label_data_df] = None
        st.session_state[__unique_labels] = None
        return

    label_df = __get_excelfile(labelfile).parse(0)

    st.session_state[__label_data_df] = label_df

    # st.session_state[__unique_labels] = {
    #     c: label_df[c].unique().sort()
    #     for c in filter_columns
    # }

    d = {}
    for c, lim in filter_columns:
        a: np.ndarray = label_df[c].unique()
        # a = ['nan' if i is np.nan else i for i in a]
        # a[np.isnan(a)] = 'nan'
        a[[i is np.nan for i in a]] = 'nan'
        a.sort()
        d[c] = (a, lim)
    st.session_state[__unique_labels] = d


@filtered_company_names_event.addListener
def __on_data_or_selected_changed():
    if not __is_key_set(__sales_data_df) or not __is_key_set(__filtered_company_names):
        return
    data_df = st.session_state[__sales_data_df]
    company_names = st.session_state[__filtered_company_names]

    data_df = data_df[data_df[filtered_column].isin(company_names)]
    data_df.rename(columns={'RepCust': 'CustomerID', 'ScNo': 'InvoiceNo', 'Qty': 'Quantity', 'ScDate': 'InvoiceDate', 'NetUPrice': 'UnitPrice', 'MarketRegion': 'Country'}, inplace=True)
    st.session_state[__filtered_df] = data_df


def __get_excelfile(file: BytesIO) -> pd.ExcelFile:
    filename = os.path.basename(file.name)

    with st.spinner(f'Loading {filename}...'):
        excel = pd.ExcelFile(file)
        return excel


def __select_sheet():
    if not __is_key_set(__sales_data_sheet_names):
        st.session_state[__sales_data_selected_sheet] = None
        return

    select_option = 'Select sheet'
    sheet_names = st.session_state[__sales_data_sheet_names]

    selected_sheet = ''

    if len(sheet_names) == 1:
        selected_sheet = st.selectbox('Select excel sheets', sheet_names)
    else:
        selected_sheet = st.selectbox('Select excel sheets', [select_option] + sheet_names)
        if selected_sheet == select_option:
            selected_sheet = None

    st.session_state[__sales_data_selected_sheet] = selected_sheet

    # if len(sheet_names) == 1:
    #     selected_sheet = st.selectbox('Select excel sheets', sheet_names)
    # else:
    #     selected_sheet = st.selectbox('Select excel sheets', [select_option] + sheet_names)
    #     if selected_sheet == select_option:
    #         selected_sheet = None

    # if selected_sheet is None:
    #     st.session_state[__sales_data_df] = None
    # else:
    #     df = dataexcelfile.parse(selected_sheet)
    #     st.session_state[__sales_data_df] = df


def __is_key_set(key):
    return key in st.session_state and st.session_state[key] is not None
