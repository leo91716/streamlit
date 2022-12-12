
from app_pages.app_page import AppPage
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import statsmodels.api as sm
import calendar
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt
import calendar

# from .df_loader import get_selected_df
from .df_loader2 import get_selected_df
# from .df_loader_set import get_selected_df
from util.font_manager import get_font

# session state key
__country_plot_fig = 'country_plot_plt'
__compare_plot_fig = 'compare_plot_fig_'
__pred_result = 'pred_result_'


class PNO_Page(AppPage):
    @staticmethod
    def _run_page():
        app()
    
    @staticmethod
    def get_name():
        return st.session_state['lang_config']['pno']['name']


def mycache(func):
    ret_d = {}

    def wrap(*args, **kwargs):
        if str(func) not in ret_d:
            ret_d[str(func)] = func(*args, **kwargs)
        return ret_d[str(func)]
    return wrap


def app():
    lang = st.session_state['lang_config']['pno']

    df, df_updated, predict_period = get_selected_df()
    df: pd.DataFrame
    # if df is not None:
    #     df.to_excel('D:/2022.07.18_UI_ML/UI-ML-V2-main/UI-ML-V2-main/tests/df2.xlsx')
    # return
    if df is None:
        return df

    if df.empty:
        st.title(lang['data_empty'])
        return

    df.rename(columns={'RepCust': 'CustomerID', 'ScNo': 'InvoiceNo', 'Qty': 'Quantity', 'ScDate': 'InvoiceDate', 'NetUPrice': 'UnitPrice', 'MarketRegion': 'Country'}, inplace=True)
    st.title(lang['data_table'])
    st.dataframe(df)

    # Country
    st.title(lang['cus vs purchase'])

    if df_updated or __country_plot_fig not in st.session_state:
        fig, ax = plt.subplots()
        plt.style.use('seaborn')
        df['Country'].value_counts().plot.bar(color='dodgerblue', ax=ax)
        for label in ax.get_xticklabels():
            label.set_fontproperties(get_font())
        st.session_state[__country_plot_fig] = fig
    fig = st.session_state[__country_plot_fig]

    st.pyplot(fig)

    c1 = st.container()
    c2 = st.container()
    show_pred(df, 2022, 6, True, False, False, c1, df_updated, predict_period, lang)
    show_pred(df, 2022, 7, False, True, True, c2, df_updated, predict_period, lang)


def show_pred(df, year, month, show_compare, show_predict_result, show_stat, base, df_updated, predict_period, lang):
    pred_key = __pred_result + f'{year}_{month}'
    if df_updated or pred_key not in st.session_state:
        df_y, df_y_hat, df_features, width, ols_reg = cal_pred(df, year, month, df_updated, predict_period)
        st.session_state[pred_key] = [df_y, df_y_hat, df_features, width, ols_reg]
    df_y, df_y_hat, df_features, width, ols_reg = st.session_state[pred_key]

    # layout
    plt_container = base.container()
    pred_container = base.container()
    stat_container = base.container()

    if show_compare:
        with plt_container:
            # slider'
            base.title(lang['actual_vs_predicted_plt'].format(year=year, month=month))
            plt_widget = base.empty()

            # plt
            plt_predict(df_y_hat, df_y, plt_widget, base, df_updated, year, month)

    if show_predict_result:
        with pred_container:
            # Custome ID form
            base.title(lang['predicti_plt'].format(year=year, month=month))
            res = pd.concat([df_features['CustomerID'], df_y_hat], axis=1, join='inner')
            base.dataframe(res)
            if base.button(lang['download_prediction']):
                filename = f'{year}-{month} Prediction'
                with st.spinner(lang['saving_prediciton'].format(filename=filename)):
                    res.to_excel(filename + '.xlsx')

    if show_stat:
        with stat_container:
            # num list
            base.title(lang['ml_stat'])
            col1, col2 = base.columns(2)
            showing_data = [
                [lang['r_squared'], ols_reg.rsquared],
                [lang['adj_r_squared'], ols_reg.rsquared_adj],
                [lang['f_statistic'], ols_reg.fvalue],
                [lang['prob'], ols_reg.f_pvalue],
                [lang['log_likeihood'], ols_reg.llf],
                [lang['aic'], ols_reg.aic],
                [lang['bic'], ols_reg.bic],
            ]
            for d in showing_data:
                col1.write(d[0])
                col2.write(round(d[1], 4))


def plt_predict(df_y_hat: pd.DataFrame, df_y: pd.DataFrame, plt_widget, base, df_updated, year, month):
    plt_key = __compare_plot_fig + f'{year}_{month}'
    if df_updated or plt_key not in st.session_state:
        figsize = (12, 6)

        fig, axs = plt.subplots(2)

        df_y_hat.plot(
            label="actual",
            marker="o",
            ax=axs[0],
            figsize=figsize,
        )  # .set_xlim(left, right)

        df_y_hat.plot(
            label="Order Volume - Actual vs Predicted",
            marker="o",
            ax=axs[1],
            figsize=figsize,
        )  # .set_xlim(left, right)

        df_y.plot(
            ax=axs[1],
            title="Order Volume - Actual vs Predicted",
            alpha=0.7,
            figsize=figsize,
            marker="o",
        )  # .set_xlim(left, right)

        st.session_state[plt_key] = fig

    fig = st.session_state[plt_key]

    plt_widget.pyplot(fig)


def cal_pred(df: pd.DataFrame, year, month, df_updated, predict_period):
    predict_month = dt.datetime(year=year, month=month, day=calendar.monthrange(year, month)[1])
    split_day = predict_period
    lastDaystr = predict_month.strftime("%Y-%m-%d")
    monthstr = predict_month.strftime("%Y-%m.0")

    # Data cleaning and transformation
    if type(df.dtypes['InvoiceDate']) is not np.datetime64:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m-%d-%y')

    df = df.loc[df['InvoiceDate'] < lastDaystr]

    # Calculate total sales
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']

    # Define invoice month
    try:
        InvoiceDateDT = pd.DatetimeIndex(df['InvoiceDate'])
        df['InvoiceMonth'] = InvoiceDateDT.to_period('M').astype(str)
        df['InvoiceMonthLabel'] = df['InvoiceMonth'] + '.' + ((InvoiceDateDT.day - 1) / split_day).astype(int).astype(str)
    except:
        df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
        df['InvoiceMonthLabel'] = df['InvoiceMonth'] + '.' + ((InvoiceDateDT.day - 1) / split_day).astype(int).astype(str)

    # Remove customer ids, which have only records for the last month
    df_now = df[df['InvoiceMonthLabel'] == monthstr]
    df_other_months = df[df['InvoiceMonthLabel'] != monthstr]

    cust_ids_nov = df_now['CustomerID'].unique().tolist()
    cust_ids_others = df_other_months['CustomerID'].unique().tolist()
    new_cust_ids = list(set(cust_ids_nov) - set(cust_ids_others))

    df = df[~df['CustomerID'].isin(new_cust_ids)]

    # Create a column to mark purchases from the UK, Germany, France and EIRE
    df['IsTriangle'] = df['Country'].apply(lambda x: 1 if x == '廣東珠三角' else 0)
    df['IsEast'] = df['Country'].apply(lambda x: 1 if x == '粵東地區' else 0)
    df['IsFu'] = df['Country'].apply(lambda x: 1 if x == '福建地區' else 0)
    df['IsWest'] = df['Country'].apply(lambda x: 1 if x == '廣西地區' else 0)

    # Feature engineering
    df_X = df[df['InvoiceMonth'] != monthstr]
    df_features = df_X.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (predict_month - x.max()).days,
        'InvoiceNo': pd.Series.nunique,
        'TotalSum': np.sum,
        'Quantity': ['mean', 'sum'],
        'IsTriangle': np.mean,
        'IsEast': np.mean,
        'IsFu': np.mean,
        'IsWest': np.mean
    }).reset_index()
    df_features.columns = [
        'CustomerID', 'Recency', 'Frequency',
        'Monetary', 'QuantityAvg', 'QuantityTotal',
        'IsTriangle', 'IsEast', 'IsFu', 'IsWest'
    ]

    cust_month_trans = pd.pivot_table(
        data=df,
        index=['CustomerID'],
        values='Quantity',
        columns=['InvoiceMonthLabel'],
        aggfunc=pd.Series.sum,
        fill_value=0
    ).reset_index()

    cust_month_trans = cust_month_trans.rename_axis('index', axis=1)

    customer_id = ['CustomerID']
    target = [monthstr]

    y = cust_month_trans[target]
    cols = [col for col in df_features.columns if col not in customer_id]
    X = df_features[cols]

    # Linear regression

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

    linreg = XGBRegressor(random_state=20, n_jobs=-1)
    # linreg = XGBRegressor(objective="reg:squarederror", n_estimators=1000, learning_rate=0.01)

    @mycache
    def get_train_pred(year, month):
        model = linreg.fit(X_train, y_train)
        # train_pred_y = model.predict(X_train)
        # test_pred_y = model.predict(X_test)
        return model  # , train_pred_y, test_pred_y
    # model, train_pred_y, test_pred_y = get_train_pred()
    model = get_train_pred(year, month)

    @st.cache
    def get_predict(year, month):
        y_hat = model.predict(X)
        df_y_hat = pd.DataFrame(y_hat)
        df_y_hat.columns = [f'{year}-{month}-predict']
        df_y = pd.DataFrame(y)
        return df_y, df_y_hat, y_hat

    df_y, df_y_hat, y_hat = get_predict(year, month)

    y_train = np.array(y_train)
    ols_reg = sm.OLS(y_train, X_train)
    ols_reg = ols_reg.fit()

    cols = [col for col in df_features.columns if col not in customer_id]
    X1 = df_features[cols]
    y1 = model.predict(X1)
    # df3=pd.DataFrame(y1)
    # df3.columns=['predict']
    # df3=pd.concat([df_features, df3], axis=1, join='inner')
    ols_reg = sm.OLS(y1, X1)
    ols_reg = ols_reg.fit()

    # cols = [col for col in df_features.columns if col not in customer_id]
    # X1 = df_features[cols]
    # ols_reg = sm.OLS(y_hat, X1)
    # ols_reg = ols_reg.fit()

    return df_y, df_y_hat, df_features, len(X.axes[0]), ols_reg
