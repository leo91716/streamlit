
import matplotlib.font_manager as font_manager
import streamlit as st


def get_font():
    if 'font' not in st.session_state:
        st.session_state['font'] = font_manager.FontProperties(fname=st.session_state['lang_config']['font_path'])
    return st.session_state['font']
