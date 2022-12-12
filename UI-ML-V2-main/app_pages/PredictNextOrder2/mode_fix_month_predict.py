
import streamlit as st


from .define import *
from util.st_util import is_key_set
from .train import retrain, show_group_result, TrainConfig
from .mode_interface import RunMode


class ModeFixMonth(RunMode):
    def __init__(self) -> None:
        super().__init__('Fix month')

    def show_options(self):
        return super().show_options()

    def run(self, do_update):
        if do_update:
            retrain(TrainConfig(fix_start_month=True, train_month_count=st.session_state[train_month_count]))

        if not is_key_set(mode_pred_next_month_cache):
            return

        show_group_result(do_update)
