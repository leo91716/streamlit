
from matplotlib import pyplot as plt
import streamlit as st
from prophet import Prophet
from dateutil.relativedelta import relativedelta
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


from .define import *
from util.st_util import is_key_set
from .mode_interface import RunMode
from .data_group import DataGroup


prophet_order = 'prophet_order'
prophet_cus_count = 'prophet_cus_count'
prophet_customers = 'prophet_customers'
prophet_df_frame = 'df_frame'
prophet_df_pred_all = 'prophet_df_pred_all'


class ModeProphet(RunMode):
    def __init__(self) -> None:
        super().__init__('Prophet')

    def show_options(self):
        st.select_slider('Predict customer count', [str(i + 1) for i in range(99)] + ['all'], '6', key=prophet_cus_count)
        st.number_input('Prophet Order', 0, 200, 20, 1, key=prophet_order)

    def run(self, do_update):
        if do_update:
            self.__update()
        self.__render()

    def __update(self):
        for igroup, dg in enumerate(st.session_state[datagroup_manager].get_data_groups()):
            dg: DataGroup
            df = dg.get_df()

            # self.separate_customer()
            df_frame = self.condense_month(df)
            st.session_state[prophet_df_frame] = df_frame
            st.dataframe(df_frame)

            df_pred_all = self.predict(df_frame)
            st.session_state[prophet_df_pred_all] = df_pred_all
        
    def generate_result(self):
        pass

    # def separate_customer(self, sales1:pd.DataFrame):
    #     def date_range(start, end):
    #         r = (end+datetime.timedelta(days=1)-start).days
    #         return [start+datetime.timedelta(days=i) for i in range(r)]

    #     customers = sales1['CustomerID'].unique().tolist()
    #     max_month = sales1['InvoiceDate'].max()
    #     min_month = sales1['InvoiceDate'].min()
    #     # min_month = datetime.datetime.strptime('2020-07-01', '%Y-%m-%d')

    #     frames = []

    #     for cus in customers:
    #         cus_df = sales1.loc[sales1['CustomerID'] == cus][['InvoiceDate', 'Quantity']]
    #         cus_df = cus_df.groupby(['InvoiceDate']).agg({'Quantity': np.sum}).reset_index()
    #         # min_month = cus_df['InvoiceDate'].min()

    #         frame = pd.DataFrame(date_range(min_month, max_month), columns=['InvoiceDate'])
    #         frame['CustomerID'] = cus
    #         frame = frame.merge(cus_df, on=['InvoiceDate'], how='left')
    #         frame = frame.fillna(0)
    #         frames.append(frame)

    #     df_frame = pd.concat(frames)
    #     return df_frame

    def condense_month(self, df: pd.DataFrame):
        df = df[['InvoiceDate', 'CustomerID', 'Quantity', ]]

        customers = df['CustomerID'].unique().tolist()

        maxday, minday = df['InvoiceDate'].max(), df['InvoiceDate'].min()
        month_dif = (maxday.year - minday.year) * 12 + maxday.month - minday.month
        minday = minday.replace(day=1)
        month_list = [minday + relativedelta(years=i // 12, months=i % 12) for i in range(month_dif + 1)]

        place_holder = pd.DataFrame({
            'InvoiceDate': [m for c in customers for m in month_list],
            'CustomerID': [c for c in customers for m in month_list],
            'Quantity': [0 for c in customers for m in month_list]
        })
        df = pd.concat([df, place_holder])

        df['year'] = df['InvoiceDate'].dt.strftime('%Y')
        df['month'] = df['InvoiceDate'].dt.strftime('%m')

        df = df.groupby(['CustomerID', 'year', 'month']).agg({
            'InvoiceDate': np.min,
            'Quantity': np.sum,
        }).reset_index()
        return df[['InvoiceDate', 'CustomerID', 'Quantity', ]]

    def predict(self, df_frame):
        pred_result_list = []
        cus_count = int(st.session_state[prophet_cus_count])

        df_frame.columns = ['ds', 'CustomerID', 'y']

        customers = list(df_frame['CustomerID'].unique())[:cus_count]
        st.session_state[prophet_customers] = customers

        bar = st.progress(0)

        for i, cus in enumerate(customers):
            bar.progress((i + 1) / cus_count)

            cus_data = df_frame[df_frame['CustomerID'] == cus]
            cus_data = cus_data[['ds', 'y', ]]

            cus_data['floor'] = 0
            cus_data['cap'] = 10000

            train = cus_data[cus_data['ds'] < cus_data['ds'].max()]
            test = cus_data
            # train = cus_data[(cus_data['ds'] >= start) & (cus_data['ds'] <= end_train)]
            # test  = cus_data[(cus_data['ds'] >= start) & (cus_data['ds'] <= end_test)]

            month_day = 30.42

            season = [(2, s) for s in [month_day * 2, month_day * 3, month_day * 5]]

            model2 = Prophet(
                growth='flat',
                changepoint_prior_scale=0.1,
                holidays_prior_scale=0.5,
                n_changepoints=100,
                interval_width=0.95,
            )
            model2.add_country_holidays(country_name='CN')
            for order, s in season:
                model2.add_seasonality(name=str(s), period=s, fourier_order=order, )
            model2.fit(train)

            pred2 = model2.predict(test)

            # model2.plot_components(pred2)

            pred2['CustomerID'] = cus
            res = pred2[['ds', 'CustomerID', 'yhat']].copy(deep=True)
            res.columns = ['ds', 'CustomerID', 'y']

            pred_result_list.append(res)

        df_pred_all = pd.concat(pred_result_list).reset_index()
        df_pred_all.columns = ['ds', 'CustomerID', 'y']

        df_pred_all['y'].loc[df_pred_all['y'] < 0] = 0

        return df_pred_all

    def plot_result(self):
        err_start = end_train - datetime.timedelta(days=14)

        df_real = df_frame_no_filter.loc[(df_frame_no_filter['ds'] > err_start) & (df_frame_no_filter['ds'] <= end_test)]
        # df_real = df_frame.loc[(df_frame['ds'] > err_start) & (df_frame['ds'] <= end_test)]
        df_real = df_real[df_real['CustomerID'].isin(customers[:cus_count])]
        df_real = df_real.sort_values(by=['CustomerID', 'ds'])
        df_pred = df_pred_all.loc[(df_pred_all['ds'] > err_start) & (df_pred_all['ds'] <= end_test)]
        df_pred = df_pred.sort_values(by=['CustomerID', 'ds'])

        for ds in df_pred['ds'].unique():
            mse, mae = self.get_error(df_real.loc[df_real['ds'] == ds]['y'], df_pred.loc[df_pred['ds'] == ds]['y'])
            # display(df_real.loc[df_real['ds'] == ds], df_pred.loc[df_pred['ds'] == ds])
            print(ds, 'mse', mse, 'mae', mae)

        row, col = 2, 3
        f, axs = plt.subplots(figsize=(14, 5), nrows=row, ncols=col)

        axlist = [axs[i][j] for i in range(row) for j in range(col)]

        start_cus = 0
        # dp = condense_month(df_pred_all)
        # df = condense_month(df_frame)

        dp = df_pred_all
        df = df_frame

        for ax, cus in zip(axlist, customers[start_cus:start_cus + row * col]):
            p = dp[dp['CustomerID'] == cus]
            r = df[df['CustomerID'] == cus]

            r.plot(kind='line', x='ds', y='y', color='green', label='Actual', ax=ax)
            p.plot(kind='line', x='ds', y='y', color='blue', label='Forecast', ax=ax)

        plt.legend()
        plt.show()

    def get_error(y1, y2):
        mse = round(np.sqrt(mean_squared_error(y1, y2)), 2)
        mae = round(mean_absolute_error(y1, y2), 2)
        return mse, mae

    def total_error(self):
        df_frame = st.session_state[prophet_df_frame]
        df_pred_all = st.session_state[prophet_df_pred_all]

        dict_error = {'ds': [], 'mse': [], 'mae': []}
        for ds in df_frame['ds'].unique():
            pass

    def __render(self):
        # mse/mae for each month
        # real/pred for each customer

        # for igroup, dg in enumerate(st.session_state[datagroup_manager].get_data_groups()):
        #     dg: DataGroup
        #     df = dg.get_df()
        # if df is not None:
        #     st.dataframe(df.loc[:, df.columns != 'InvoiceMonth'])
        #     st.write(df.shape)
        #     st.write(df['CustomerID'].nunique())
        self.plot_result()
        df_real_month = condense_month(df_real)
        df_pred_month = condense_month(df_pred)

        for ds in df_pred_month['ds'].unique():
            mse, mae = get_error(df_real_month.loc[df_real_month['ds'] == ds]['y'], df_pred_month.loc[df_pred_month['ds'] == ds]['y'])
            print(ds, 'mse', mse, 'mae', mae)

        r = condense_month(df_real_month)
        p = condense_month(df_pred_month)

        r['pred'] = p['y']
        display(r)
