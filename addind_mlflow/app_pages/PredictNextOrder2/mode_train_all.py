
import streamlit as st


from .define import *
from util.st_util import is_key_set
from .train import retrain, show_group_result, TrainConfig
from .mode_interface import RunMode


class ModeTrainAll(RunMode):
    def __init__(self) -> None:
        super().__init__('Train all')

    def show_options(self):
        # st.write('Train time')
        pass

    def run(self, do_update):
        if do_update:
            retrain(TrainConfig(fix_start_month=False, train_month_count=st.session_state[train_month_count], do_combine_last_month=True, last_month='2022-04'))

        if not is_key_set(mode_pred_next_month_cache):
            return

        show_group_result(do_update)
