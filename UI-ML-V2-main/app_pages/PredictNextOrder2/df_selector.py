
import numpy as np
import os
from io import BytesIO
import streamlit as st
import pandas as pd

from util.time_util import measure_time
from util.st_util import is_key_set
from .data_group import DataGroupManager, DataGroup
from util.load_data import load_excel
from .define import *


def select_df(lang):
    __load_SC(lang)
    __load_filter(lang)

    if is_key_set(raw_sc_df) and is_key_set(raw_filter_df):
        __set_datagroup(lang)


def __load_SC(lang):
    # st.sidebar.file_uploader("SC", type=['xlsx'], help='upload', on_change=__on_SC_changed, key=SC_file, args=(lang, ))

    if not is_key_set(SC_file):
        # filename = '../data/pno/SC_test.XLSX'
        # with open(filename, 'rb') as f:
        #     st.session_state[SC_file] = BytesIO(f.read())

        st.session_state[SC_file] = ''
        __on_SC_changed(lang)

    if not is_key_set(SC_file):
        st.warning(lang['no_sc'])
    else:
        pass
        # st.write(f'SC top 100 out of {st.session_state[__raw_sc_df].shape[0]}')
        # st.dataframe(st.session_state[__raw_sc_df].head(100))


def __on_SC_changed(lang):
    # file: BytesIO = st.session_state[SC_file]

    with st.spinner(lang['loading_sc']):
        # sheet_names = pd.ExcelFile(file).sheet_names

        # df = load_excel(file, sheet_names)
        # df.to_csv('SC_test.csv')

        df = pd.read_csv('../data/pno/SC_all_201807_202207.csv')

        df.rename(columns={'RepCust': 'CustomerID', 'ScNo': 'InvoiceNo', 'Qty': 'Quantity', 'ScDate': 'InvoiceDate', 'NetUPrice': 'UnitPrice', 'MarketRegion': 'Country'}, inplace=True)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        st.session_state[raw_sc_df] = df


def __load_filter(lang):
    # st.sidebar.file_uploader("filter", type=['xlsx'], help='upload', on_change=__on_filter_changed, key=filter_file, args=(lang, ))

    if not is_key_set(filter_file):
        # filename = '../data/pno/客戶標籤 (1).xlsx'
        # with open(filename, 'rb') as f:
        #     st.session_state[filter_file] = BytesIO(f.read())
        st.session_state[filter_file] = ''
        __on_filter_changed(lang)

    if not is_key_set(filter_file):
        st.warning(lang['no_filter'])
    else:
        pass
        # st.write(f'Filter top 100 out of {st.session_state[__raw_filter_df].shape[0]}')
        # st.dataframe(st.session_state[__raw_filter_df].head(100))


def __on_filter_changed(lang):
    # file: BytesIO = st.session_state[filter_file]

    # with st.spinner(f'Loading filter...'):
    #     df = load_excel(file)
    #     st.session_state[raw_filter_df] = df
    # df.to_csv('SC_filter.csv')

    df = pd.read_csv('../data/pno/Filter_test.csv')
    st.session_state[raw_filter_df] = df

    st.session_state[filter_dict] = {
        f: df[f].unique()
        for f in filter_col
    }


def __set_datagroup(lang):
    if datagroup_manager not in st.session_state:
        st.session_state[datagroup_manager] = __create_default_datagroup_manager(lang)
    dm: DataGroupManager = st.session_state[datagroup_manager]

    st.title(lang['data_group_config'])

    __datagroup_filter(dm, lang)


def __datagroup_filter(dm: DataGroupManager, lang):
    name = st.text_input(lang['create_new_group'])
    if st.button(lang['create_new_group_button']) and name != '':
        dm.create_new_group(name)

    # dg:DataGroup = st.selectbox('Delete tab', dm.get_data_groups(), 0, format_func=lambda x: x.get_displayname())
    # if st.button('delete'):
    #     # print(dg.get_displayname(), dg.get_id())
    #     if len(dm.get_data_groups()) > 1:
    #         dm.remove_group(dg.get_id())
    #         st.experimental_rerun()

    tabs = st.tabs(dm.get_displaynames())
    for tab, datagroup in zip(tabs, dm.get_all_data_groups()):
        with tab:
            datagroup.show_tab(st.session_state[filter_dict], st.session_state[raw_sc_df], st.session_state[raw_filter_df])
    st.markdown('----')


def __create_default_datagroup_manager(lang):
    dm = DataGroupManager(lang)
    dm.create_new_group('All').set_filter({
        'BU': [
            'MABU', 'REBU', 'RNBU', 'RSBU', 'RWBU', 'TWBU', 
        ],
        'ProdGroup': [
            'PA', 'PZ', 'PVC'
        ],
        'MonthGrade': [
            'E', 'C', 'A', 'G', 'F', 'D', 'H', 'B'
        ],
        'QtyGrade': [
            'C', 'D', 'B', 'A'
        ],
        'NetUPriceGrade': [
            'A', 'C', 'B'
        ]
    }, st.session_state[filter_dict]).set_group_count(1).set_do_check_contain(False).\
        set_rfm([1,2,3,4,5], [5], [1,2,3,4,5]).set_selected_month('2020-06', '2022-07')

    # dm = DataGroupManager(lang)
    # dm.create_new_group('RSBU').set_filter({
    #     'BU': [
    #         'RSBU',
    #     ],
    #     'ProdGroup': [
    #         'PA', 'PZ', 'PVC'
    #     ],
    #     'MonthGrade': [
    #         'E', 'C', 'A', 'G', 'F', 'D', 'H', 'B'
    #     ],
    #     'QtyGrade': [
    #         'C', 'D', 'B', 'A'
    #     ],
    #     'NetUPriceGrade': [
    #         'A', 'C', 'B'
    #     ]
    # }, st.session_state[filter_dict]).set_group_count(1)

    # dm.create_new_group('REBU').set_filter({
    #     'BU': [
    #         'REBU'
    #     ],
    #     'ProdGroup': [
    #         'PA', 'PZ', 'PVC'
    #     ],
    #     'MonthGrade': [
    #         'E', 'C', 'A', 'G', 'F', 'D', 'H', 'B'
    #     ],
    #     'QtyGrade': [
    #         'C', 'D', 'B', 'A'
    #     ],
    #     'NetUPriceGrade': [
    #         'A', 'C', 'B'
    #     ]
    # }, st.session_state[filter_dict]).set_group_count(1)

    # dm.create_new_group('all').set_filter({
    #     'BU': [
    #         'MABU', 'REBU', 'RNBU', 'RSBU', 'RWBU', 'TWBU'
    #     ],
    #     'ProdGroup': [
    #         'PA', 'PZ', 'PVC'
    #     ],
    #     'MonthGrade': [
    #         'E', 'C', 'A', 'G', 'F', 'D', 'H', 'B'
    #     ],
    #     'QtyGrade': [
    #         'C', 'D', 'B', 'A'
    #     ],
    #     'NetUPriceGrade': [
    #         'A', 'C', 'B'
    #     ]
    # }, st.session_state[filter_dict]).set_group_count(3)

    return dm

# full filter
# {
#     'BU': [
#         'MABU', 'REBU', 'RNBU', 'RSBU', 'RWBU', 'TWBU'
#     ],
#     'ProdGroup': [
#         'PA', 'PZ', 'PVC'
#     ],
#     'MonthGrade': [
#         'E', 'C', 'A', 'G', 'F', 'D', 'H', 'B'
#     ],
#     'QtyGrade': [
#         'C', 'D', 'B', 'A'
#     ],
#     'NetUPriceGrade': [
#         'A', np.nan, 'C', 'D', 'B'
#     ]
# }
