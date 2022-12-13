"""Frameworks for running multiple Streamlit applications as a single app."""
import streamlit as st

from app_pages.app_page import AppPage


class MultiApp:
    """Framework for combining multiple streamlit applications."""

    def __init__(self):
        self.apps = []

    def add_app(self, page: AppPage):
        self.apps.append({
            "page": page,
        })

    def run(self):
        def f(app):
            name = app['page'].get_name()
            return name

        app = st.sidebar.radio(
            st.session_state['lang_config']['multi_app']['menu_title'],
            self.apps,
            format_func=f,)
            # index=6,)

        # app = st.sidebar.radio(
        #     '功能',
        #     self.apps,
        #     format_func=lambda app: app['page'].get_name())

        app['page'].run()
