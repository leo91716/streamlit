
import streamlit as st

from app_pages.app_page import AppPage
from util.config_setup import *


class ConfigPage(AppPage):
    @staticmethod
    def _run_page():
        lang = st.session_state['lang_config']['config']
        st.title(lang['title'])
        lang_files = get_lang_files()
        lang_filename = st.selectbox(lang['language'], lang_files, lang_files.index(st.session_state['current_lang']))

        if st.button(lang['submit']):
            set_lang(lang_filename)
            st.experimental_rerun()
    
    @staticmethod
    def get_name():
        return st.session_state['lang_config']['config']['name']
