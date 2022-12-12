
import streamlit as st


from .data_group import DataGroup


#### keys ####
SC_file = 'pno2_SC_file'
filter_file = 'pno2_filter_file'

raw_sc_df = 'pno2_raw_sc'
raw_filter_df = 'pno2_raw_filter'
filter_dict = 'pno2_filter_dict'

datagroup_manager = 'pno2_datagroup_manager'

run_mode_key = 'pno2_run_mode'

train_month_count = 'train_month_count'


#### mode cache ####
mode_pred_next_month_cache = 'pred_next_month_cache'


#### options ####
#columns = ['BU', 'Year', 'Month', 'ProdGroup', 'RepCust', 'MonthGrade', 'QtyGrade', 'NetUPriceGrade']
filter_col = ['BU', 'ProdGroup', 'MonthGrade', 'QtyGrade', 'NetUPriceGrade']
filter_target = 'RepCust'

#### pred next keys ####
pn_single_error = 'pn_single_error'
pn_single_rsquare = 'pn_single_rsquare'
pn_single_predict = 'pn_single_predict'
pn_rsquare = 'pn_rsquare'

pn_error_changed = 'pn_error_changed'
pn_error = 'pn_error'
pn_error_select = 'pn_error_select'

#### Data types ####

class PredictResult:
    def __init__(self) -> None:
        self.data_group: DataGroup = None
        self.pred_month = []
        self.df_pred_x = []
        self.df_pred_y = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.model = []
        self.train_pred_y = []
        self.test_pred_y = []
        self.pred_y = []
        self.mse_train = []
        self.mae_train = []
        self.mse_test = []
        self.mae_test = []
        self.mse_pred = []
        self.mae_pred = []
        self.rsquared = []


class PredictGroup:
    def __init__(self) -> None:
        self.data_group: DataGroup = None
        self.result_dict = None



