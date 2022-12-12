
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import streamlit as st

from util.st_util import is_key_set


dg_sc_month = 'dg_sc_month'


class DataGroupManager:
    def __init__(self, lang) -> None:
        self.__id_count = 0
        self.__data_group = []
        self.__data_group_creator = None
        self.__data_group_dict = {}
        self.__lang = lang

        # self.__add_group_creator()

    def create_new_group(self, displayname):
        group = DataGroup(self.__id_count, displayname, self, self.__lang)
        self.__data_group.append(group)
        self.__data_group_dict[group.get_id()] = group
        self.__id_count += 1
        return group

    # def __add_group_creator(self):
    #     self.__data_group_creator = DataGroupCreator(self.__id_count, 'Create new')
    #     self.__id_count += 1

    def get_displaynames(self):
        return [d.get_displayname() for d in self.__data_group]

    def get_data_groups(self):
        return self.__data_group

    def get_all_data_groups(self):
        return self.__data_group + ([self.__data_group_creator] if self.__data_group_creator is not None else [])

    def remove_group(self, id):
        if len(self.__data_group) <= 1:
            return
        if id in self.__data_group_dict:
            group = self.__data_group_dict.pop(id)
            self.__data_group.remove(group)


class DataGroup:
    def __init__(self, id, displayname, manager: DataGroupManager, lang) -> None:
        self.__id = id
        self.__displayname = displayname
        self.__manager = manager
        self.__lang = lang
        self.__filter = None
        self.__default_filter = None
        self.__default_group_count = 1
        self.__group_count = 1
        self.__df = None
        self.__month_format = '%Y-%m'

        self.__selected_month = None
        self.__default_selected_month = None
        self.__do_check_contain = False
        self.__default_do_check_contain = False

        self.__rfm_range = 5
        self.__r = self.__get_rfm_default()
        self.__f = self.__get_rfm_default()
        self.__m = self.__get_rfm_default()
        self.__default_r = self.__get_rfm_default()
        self.__default_f = self.__get_rfm_default()
        self.__default_m = self.__get_rfm_default()

    def __get_rfm_default(self):
        return [i for i in range(1, self.__rfm_range + 1, 1)]

    def __eq__(self, __o: object) -> bool:
        if not type(__o) == type(self):
            r = True
        else:
            r = (self.__id == __o.__id)
        return r

    def get_displayname(self):
        return self.__displayname

    def show_tab(self, filter: dict, df_sc: pd.DataFrame, df_filter: pd.DataFrame):
        # rename
        display_name = st.text_input(self.__lang['group_name'], self.__displayname, key=f'change_name_key_{self.get_id()}')
        if display_name != self.__displayname:
            self.__displayname = display_name
            st.experimental_rerun()

        # delete
        if st.button(self.__lang['delete_group'], key=f'data_group_delete_{self.get_id()}'):
            self.__manager.remove_group(self.get_id())
            st.experimental_rerun()

        # filter
        st.subheader(self.__lang['filter'])
        if self.__filter is None:
            self.__filter = filter.copy()
            self.__default_filter = filter.copy()

        for k, v in filter.items():
            key = f'datagroup_{self.__id}_{k}'
            self.__filter[k] = st.multiselect(k, v, self.__default_filter[k], key=key)

        # group
        st.subheader(self.__lang['group_group'])
        self.__group_count = st.slider(self.__lang['group_group_count'], 1, 8, self.__default_group_count, 1, key=f'datagroup_{self.__id}_groupcount')

        # select time frame
        st.subheader('Select time frame')
        months = self.__get_sc_months(df_sc)
        if self.__default_selected_month is None:
            self.__default_selected_month = (months[0], months[-1])
        self.__selected_month = st.select_slider('Time frame', months, self.__default_selected_month, key=f'datagroup_{self.__id}_selected_month')
        self.__do_check_contain = st.checkbox('Check transaction contain time frame', self.__default_do_check_contain, key=f'datagroup_{self.__id}_check_contain')

        # rfm filter
        st.subheader('RFM filter')
        self.__r = st.multiselect('Recency', self.__get_rfm_default(), self.__default_r, key=f'datagroup_{self.__id}_r')
        self.__f = st.multiselect('Frequency', self.__get_rfm_default(), self.__default_f, key=f'datagroup_{self.__id}_f')
        self.__m = st.multiselect('Monetary', self.__get_rfm_default(), self.__default_m, key=f'datagroup_{self.__id}_m')

    def get_start_month(self):
        return datetime.strptime(self.__selected_month[0], self.__month_format)

    def get_train_end_month(self):
        return datetime.strptime(self.__selected_month[1], self.__month_format) + relativedelta(month=1) - relativedelta(day=1)

    def get_predict_end_month(self):
        return self.get_train_end_month() + relativedelta(month=2) - relativedelta(day=1)

    def __get_sc_months(self, df_sc):
        if not is_key_set(dg_sc_month):
            months = df_sc['InvoiceDate'].dt.strftime("%Y-%m").unique()
            months.sort()
            months = list(months)
            months.append((df_sc['InvoiceDate'].max() + relativedelta(months=1)).strftime("%Y-%m"))
            st.session_state[dg_sc_month] = months
        return st.session_state[dg_sc_month]

    def submit(self, df_sc: pd.DataFrame, df_filter: pd.DataFrame):
        self.__run_filter_group(df_sc, df_filter)

    def get_df(self) -> pd.DataFrame:
        return self.__df

    def __run_filter_group(self, df_sc: pd.DataFrame, df_filter: pd.DataFrame):
        # filter
        filtered_df = df_filter
        for k, v in self.__filter.items():
            filtered_df = filtered_df.loc[filtered_df[k].isin(v)]
        companies = filtered_df['RepCust'].unique()

        df = df_sc.loc[df_sc['CustomerID'].isin(companies)]
        df = df.loc[df['BU'].isin(filtered_df['BU'])]

        df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
        df['InvoiceMonthStr'] = df['InvoiceMonth'].astype(str)

        if self.__do_check_contain:
            df = self.__check_contain(df)

        df = df.loc[(df['InvoiceMonthStr'] >= self.__selected_month[0]) & (df['InvoiceMonthStr'] <= self.__selected_month[1])]

        df = self.__check_rfm(df)

        self.__df = df

        # st.write(self.__df.astype('object'))

    def __check_contain(self, df: pd.DataFrame):
        a = df['CustomerID'].nunique()
        start_month = self.get_start_month().strftime('%Y-%m')
        end_month = self.get_train_end_month().strftime('%Y-%m')

        def check_start(d):
            return d <= start_month

        def check_end(d):
            return d >= end_month

        customers = df.groupby('CustomerID').agg({
            'InvoiceMonthStr': [check_start, check_end],
        }).reset_index()['CustomerID']

        df = df.loc[df['CustomerID'].isin(customers)]
        b = df['CustomerID'].nunique()
        return df

    def __check_rfm(self, df):
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda date: (self.get_predict_end_month() - date.max()).days,
            'InvoiceNo': lambda num: len(num),
            'Quantity': lambda price: price.sum(),
        })

        rfm.columns = ['recency', 'frequency', 'monetary']
        rfm['recency'] = rfm['recency'].astype(int)
        rfm['r_quartile'] = pd.qcut(rfm['recency'], 5, [1, 2, 3, 4, 5])
        rfm['f_quartile'] = pd.qcut(rfm['frequency'], 5, [1, 2, 3, 4, 5])
        rfm['m_quartile'] = pd.qcut(rfm['monetary'], 5, [1, 2, 3, 4, 5])

        rfm = rfm.loc[rfm['r_quartile'].isin(self.__r)]
        rfm = rfm.loc[rfm['f_quartile'].isin(self.__f)]
        rfm = rfm.loc[rfm['m_quartile'].isin(self.__m)]

        rfm = rfm.reset_index()

        customers = rfm['CustomerID']

        df = df.loc[df['CustomerID'].isin(customers)]

        return df

    def set_filter(self, default_filter, filter):
        for k in filter:
            sd = set(default_filter[k])
            sf = set(filter[k])
            default_filter[k] = list(sd - (sd - sf))

        self.__filter = default_filter.copy()
        self.__default_filter = default_filter.copy()
        return self

    def set_group_count(self, count):
        self.__group_count = count
        self.__default_group_count = count
        return self

    def get_group_count(self):
        return self.__group_count

    def get_id(self):
        return self.__id

    def set_selected_month(self, start_month, end_month):
        self.__default_selected_month = [start_month, end_month]
        return self

    def set_do_check_contain(self, do_check_contain):
        self.__default_do_check_contain = do_check_contain
        return self

    def set_rfm(self, r, f, m):
        self.__default_r = r
        self.__default_f = f
        self.__default_m = m
        return self


class DataGroupCreator(DataGroup):
    def __init__(self, id, displayname) -> None:
        self.__id = id
        self.__displayname = displayname
        self.__filter = None
        self.__default_filter = None
        self.__group_count = 1
        self.__df = None

    def get_displayname(self):
        return self.__displayname

    def show_tab(self, filter: dict, df_sc: pd.DataFrame, df_filter: pd.DataFrame):
        display_name = st.text_input('New ', self.__displayname)
        if display_name != self.__displayname:
            self.__displayname = display_name
            # st.experimental_rerun()

    def submit(self, df_sc: pd.DataFrame, df_filter: pd.DataFrame):
        pass

    def get_df(self):
        return self.__df

    def set_filter(self, default_filter, filter):
        return self

    def set_group_count(self, count):
        self.__group_count = count
        return self

    def get_group_count(self):
        return self.__group_count

    def get_id(self):
        return self.__id
