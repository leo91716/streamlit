
import numpy as np
import pandas as pd
# from pydantic.errors import NoneIsAllowedError
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from app_pages.app_page import AppPage


class EDA3Page(AppPage):
    @staticmethod
    def _run_page():
        app()

    @staticmethod
    def get_name():
        return st.session_state['lang_config']['eda3']['name']


def app():
    lang = st.session_state['lang_config']['eda3']

    # function for getting wider space on web page
    def _max_width_():
        max_width_str = f"max-width: 950px;"
        st.markdown(
            f"""
                <style>
                .reportview-container .main .block-container{{
                    {max_width_str}
                }}
                </style>    
                """,
            unsafe_allow_html=True,
        )
    # calling the function for full page
    _max_width_()

    # writng some on header part
    st.write(f"<h2 style='text-align: center;'>{lang['title']}</h2>", unsafe_allow_html=True)
    # Web App Title
    # st.markdown('''
    # (https://github.com/pandas-profiling/pandas-profiling).
    # ''')

    # asking for file
    file_upload = st.sidebar.file_uploader(lang['upload_here'], type=['csv'], help=lang['upload_help'])
    name = st.sidebar.selectbox(lang['select_sample_data'], options=['None', 'Forbes Richest Atheletes', 'IT Salary Survey EU 2020'], help=lang['select_sample_help'])

    # smple file getting function
    def get_dataset(name, sample=True, custome=False):
        try:
            if sample:
                if name == 'Forbes Richest Atheletes':  # matchin user choose file
                    df = pd.read_csv('app_pages\eda_pd_profiling\ForbesRichestAtheletes.csv')
                    return df  # retruning the data frame
                elif name == 'IT Salary Survey EU 2020':
                    df = pd.read_csv('app_pages\eda_pd_profiling\ITSalarySurveyEU2020.csv')
                    return df
            if custome:
                df = pd.read_csv(file_upload)
                return df
        except Exception:
            print('error n load_dataset')

    # giving result by choosing dataset, custome or sample
    if file_upload is None:
        df = get_dataset(name=name)
    else:
        df = get_dataset(name=name, custome=True, sample=False)

    if df is not None:
        if st.sidebar.checkbox(lang['show_data'], value=True):
            st.write()

            st.dataframe(df)
    else:
        st.warning(lang['hint1'])

    if df is not None:
        if st.sidebar.button(lang['create_report'], help=lang['create_report_help']):
            pr = ProfileReport(df)
            # pr = ProfileReport(df, explorative=True)
            st.markdown(f"<h2 style='text-align: center;'>{lang['generating_report']}</h2>", unsafe_allow_html=True)
            st_profile_report(pr)
    else:
        st.warning(lang['hint2'])
