
import streamlit as st
from io import BytesIO
import pandas as pd
import os
import enum


class FileType(enum.Enum):
    DATA_FILE = '0'
    LABEL_FILE = '1'


filter_columns = ['MonthGrade', 'QtyGrade', 'NetUPriceGrade']
filtered_column = 'RepCust'


def get_selected_df():
    '''
    Place file uploader at side bar and load selected excel file
    If file has more than one sheets, place selection box to selct loaded sheet
    Returns DataFrame if loads a sheet, otherwise None
    '''
    data_df = __getdatadf()

    label_df = __getlabels()

    if label_df is not None:
        filtered_column_names = __select_filters(label_df)

    if data_df is None or label_df is None or filtered_column_names is None:
        return None

    data_df = data_df[data_df[filtered_column].isin(filtered_column_names)]

    data_df.rename(columns={'RepCust': 'CustomerID', 'ScNo': 'InvoiceNo', 'Qty': 'Quantity', 'ScDate': 'InvoiceDate', 'NetUPrice': 'UnitPrice', 'MarketRegion': 'Country'}, inplace=True)
    st.dataframe(data_df)
    return data_df


def __getdatadf():
    datafile = st.sidebar.file_uploader('Upload Sales Data Here', type=['xlsx'], help="Only `xlsx` Please", key=FileType.DATA_FILE)
    if datafile is None:
        st.write("Please upload sales data")
        return None

    sheetname = __select_sheet(FileType.DATA_FILE, datafile)
    if sheetname is None:
        return None

    df = __get_df(FileType.DATA_FILE, datafile, sheetname)

    return df


def __getlabels():
    labelfile = st.sidebar.file_uploader('Upload Customer Labels Here', type=['xlsx'], help="Only `xlsx` Please", key=FileType.LABEL_FILE)
    if labelfile is None:
        st.write("Please upload customer label")
        return

    df = __get_df(FileType.LABEL_FILE, labelfile)

    return df


def __select_filters(label_df: pd.DataFrame):
    unique_labels = {
        c: label_df[c].unique()
        for c in filter_columns
    }

    st.write("Please select filter")
    selected = {k: st.multiselect(k, v, v) for k, v in unique_labels.items()}

    if not st.button('Submit'):
        return None

    df = label_df
    for k, v in selected.items():
        df = df[df[k].isin(v)]
    return df[filtered_column]


@st.cache
def __get_df(fileType: FileType, file, sheet_name=0) -> pd.DataFrame:
    kwargs = {}
    # if fileType == FileType.LABEL_FILE:
    #     kwargs['skiprows'] = 1
    df = __get_excelfile(fileType, file).parse(sheet_name, skiprows=1)
    return df


def __get_excelfile(fileType: FileType, file: BytesIO) -> pd.ExcelFile:
    filename = os.path.basename(file.name)
    key_filename = f'pno_filename_{fileType}'
    key_file = f'pno_file_{fileType}'
    if 'pno_filename' in st.session_state and st.session_state[key_filename] == filename:
        return st.session_state[key_file]

    with st.spinner(f'Loading {filename}...'):
        excel = pd.ExcelFile(file)
        st.session_state[key_filename] = filename

        st.session_state[key_file] = excel
        return excel


def __select_sheet(fileType: FileType, file):
    select_option = 'Select sheet'
    sheet_names = __get_excelfile(fileType, file).sheet_names

    if len(sheet_names) == 1:
        return sheet_names[0]

    selected = st.selectbox('Select excel sheets', [select_option] + sheet_names)
    if selected == select_option:
        return None
    return selected
