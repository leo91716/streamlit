
import streamlit as st
from PIL import Image

from app_pages.multiapp import MultiApp
from util.config_setup import config_setup


def run():
    config_setup()

    st.set_page_config(page_title=st.session_state['lang_config']['root']['page_title'], page_icon="resources/mitac-logo.png", initial_sidebar_state='auto')  # , layout = 'wide')

    logo = Image.open(r'resources/mitac-logo.png')
    st.sidebar.image(logo, width=120)

    run_app()

def run_app():

    app = MultiApp()

    from app_pages.home import HomePage
    app.add_app(HomePage)

    from app_pages.eda_pd_profiling.eda3 import EDA3Page
    app.add_app(EDA3Page)

    from app_pages.decision1 import Decision1Page
    app.add_app(Decision1Page)

    from app_pages.Classification.clf1 import CLF1Page
    app.add_app(CLF1Page)

    from app_pages.Regression.mlv2_V2 import MLV2_V2_Page
    app.add_app(MLV2_V2_Page)

    # from app_pages.PredictNextOrder.pno import PNO_Page
    # app.add_app(PNO_Page)

    from app_pages.PredictNextOrder2.pno import PNO2_Page
    app.add_app(PNO2_Page)

    from app_pages.config_page import ConfigPage
    app.add_app(ConfigPage)

    app.run()


if __name__ == '__main__':
    run()
