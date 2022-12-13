
import streamlit as st

from ..app_page import AppPage
from .df_selector import select_df
from .define import *
from util.st_util import is_key_set


from .mode_predict_next_month import ModePredictNext
from .mode_fix_month_predict import ModeFixMonth
from .mode_train_all import ModeTrainAll
from .mode_interface import RunMode
from .mode_prophet import ModeProphet

__process_mode = {
    # 'test': test_run,
    'prophet': ModeProphet(),
    'predict next month': ModePredictNext(),
    'predict fix month': ModeFixMonth(),
    'train all': ModeTrainAll(),
    
}


'''
side bar:
    Select df

tabs:
    select/add group
    default
'''


class PNO2_Page(AppPage):
    @staticmethod
    def _run_page():
        app()

    @staticmethod
    def get_name():
        return st.session_state['lang_config']['pno2']['name']


def app():
    lang = st.session_state['lang_config']['pno2']
    st.session_state['page_lang'] = lang

    select_df(lang)

    if is_key_set(datagroup_manager):
        dm = st.session_state[datagroup_manager]

        st.title(lang['train_configuration'])

        st.slider(lang['training_month_count'], 6, 18, 12, step=1, key=train_month_count)
        st.selectbox(lang['analyze_mode'], __process_mode.keys(), 0, key=run_mode_key)

        run_mode: RunMode = __process_mode[st.session_state[run_mode_key]]

        run_mode.show_options()

        do_update = st.button(lang['start_analyze'])
        if do_update:
            for datagroup in dm.get_data_groups():
                datagroup.submit(st.session_state[raw_sc_df], st.session_state[raw_filter_df])

        run_mode.run(do_update)
