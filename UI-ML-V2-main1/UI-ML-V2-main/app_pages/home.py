
import streamlit as st

from app_pages.app_page import AppPage


class HomePage(AppPage):
    @staticmethod
    def _run_page():
        app()

    @staticmethod
    def get_name():
        return st.session_state['lang_config']['home']['name']


def app():
    lang = st.session_state['lang_config']['home']
    # function that will change the width of the web page

    def _max_width_():
        max_width_str = f"max-width: 800px;"
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
    # _max_width_()
    """
    html = 
    <style>
        /* Disable overlay (fullscreen mode) buttons */
        .overlayBtn {
            display: none;
        }

        /* Remove horizontal scroll */
        .element-container {
            width: auto !important;
        }

        .fullScreenFrame > div {
            width: auto !important;
        }

        /* 2nd thumbnail */
        .element-container:nth-child(4) {
            top: -266px;
            left: 350px;
        }

        /* 1st button */
        .element-container:nth-child(3) {
            left: 10px;
            top: -60px;
        }

        /* 2nd button */
        .element-container:nth-child(5) {
            left: 360px;
            top: -326px;
        }
    </style>
"""
    st.write(f"<h1 style='text-align: center;'>{lang['title']}</h1>", unsafe_allow_html=True)
    # JUST TAKING FOR SOME SPACE
    st.markdown("<h1 style='text-align: center;'></h1>", unsafe_allow_html=True)
    #st.markdown(html, unsafe_allow_html=True)

    st.image("resources/main.jpg", width=1000)
    #st.button("Show", key=1)

    #st.image("https://www.w3schools.com/howto/img_forest.jpg", width=300)
    #st.button("Show", key=2)
    #st.markdown("<h1 style='text-align: center;'>🤖 UI-ML 🤖™</h1>", unsafe_allow_html=True)

    #st.write("<h5 style='text-align: center;'>🎈Create Machine Learning Model by just `Clicking`🎈</h5>", unsafe_allow_html=True)
    # JUST TAKING FOR SOME SPACE
    st.markdown("<h1 style='text-align: left;'></h1>", unsafe_allow_html=True)

    # writing about classification
    #clf = """ 請選擇左列選項"""
    st.write(f"<h2 style='text-align: left;'>{lang['select_left']}</h2>", unsafe_allow_html=True)
    # st.write(clf)
