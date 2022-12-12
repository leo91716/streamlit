
import streamlit as st


def is_key_set(key):
    return key in st.session_state and st.session_state[key] is not None
