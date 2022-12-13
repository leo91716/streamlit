
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from dateutil.relativedelta import relativedelta
import streamlit as st
from datetime import datetime
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import pandas as pd
import io
import matplotlib.pyplot as plt


from util.st_util import is_key_set
from .define import *
from .data_group import DataGroup


class TrainConfig:
    def __init__(self, fix_start_month=False, train_month_count=12, do_combine_last_month=False, last_month='') -> None:
        self.fix_start_month = fix_start_month
        self.train_month_count = train_month_count
        self.do_combine_last_month = do_combine_last_month
        self.combine_last_month = last_month


def retrain(config: TrainConfig):
    group_result = []
    st.session_state[mode_pred_next_month_cache] = group_result

    progress = st.progress(0.0)
    group_count = len(st.session_state[datagroup_manager].get_data_groups())

    for igroup, dg in enumerate(st.session_state[datagroup_manager].get_data_groups()):
        pr2, months = __extract_feature(progress, group_result, config, igroup, group_count, dg)

        if config.do_combine_last_month:
            __train_combine(pr2, months, config.combine_last_month)
        else:
            __train(pr2, months)
        __predict(pr2, months)


def __extract_feature(progress, group_result, config: TrainConfig, igroup, group_count, dg):
    dg: DataGroup
    df = dg.get_df()

    # country
    country_names = ['廣東珠三角', '台灣', '蘇州地區', '無錫地區', '常州地區', '浙北', '南通地區', '粵東地區', '福建地區', '廣西地區', ]
    country_types = ['IsTrangle', 'IsTW', 'IsSu', 'IsWu', 'IsCha', 'IsChu', 'IsNan', 'IsEast', 'IsFu', 'IsWest', ]

    for cname, ctype in zip(country_names, country_types):
        df[ctype] = df['Country'].apply(lambda x: 1 if x == cname else 0)
    df['IsOther'] = df['Country'].apply(lambda x: 1 if x not in country_names else 0)

    country_names += ['Other']
    country_types += ['IsOther']

    # add features
    df['Monetary'] = df['UnitPrice'] * df['Quantity']
    currency_table = {
        'RMB': 0.14,
        'NTD': 0.33,
        'USD': 1,
        'MYR': 0.22,
        'THB': 0.27,
        'VND': 0.000043,
    }
    df['USPrice'] = df['Cur'].map(currency_table) * df['UnitPrice']

    # generate months
    date_format = '%Y-%m'

    first_month = datetime.strptime('2020-07', date_format)
    last_month = datetime.strptime('2022-07', date_format)

    # first_month = datetime.strptime('2020-07', date_format)
    # first_month = datetime.strptime('2021-07', date_format)
    # last_month = datetime.strptime('2022-07', date_format)

    start_month = add_month(first_month, config.train_month_count)

    # ** fix/no fix start month
    print(first_month, start_month, last_month)
    months = generate_months(first_month, start_month, last_month, date_format, config.fix_start_month)

    pr2 = PredictResult()
    pr2.data_group = dg
    group_result.append(pr2)
    for i, m in enumerate(months):
        dg = pr2.data_group

        pr2.pred_month.append(m[3])

        progress.progress((i + 1 + len(months) * igroup) / len(months) / group_count)

        # split data
        df_model_x, df_model_data = get_df_x(df, country_types, m[1], m[0])
        df_model_y = get_df_y(df, m[1], m[0], df_model_x)

        customerid = df_model_x['CustomerID'].unique()

        df_pred_x, df_analyze_data = get_df_x(df, country_types, m[3], m[2])
        df_pred_x = df_pred_x.loc[df_pred_x['CustomerID'].isin(customerid)]
        df_analyze_data = df_analyze_data.loc[df_analyze_data['CustomerID'].isin(customerid)]
        df_pred_y = get_df_y(df, m[3], m[2], df_model_x)

        cus = set(df_model_x['CustomerID']) & set(df_pred_x['CustomerID'])

        df_model_x = df_model_x.loc[df_model_x['CustomerID'].isin(cus)]
        df_model_y = df_model_y.loc[df_model_y['CustomerID'].isin(cus)]
        df_pred_x = df_pred_x.loc[df_pred_x['CustomerID'].isin(cus)]
        df_pred_y = df_pred_y.loc[df_pred_y['CustomerID'].isin(cus)]

        # group
        if dg.get_group_count() != 1:
            kmeans = KMeans(
                init="random",
                n_clusters=dg.get_group_count(),
                n_init=10,
                max_iter=300,
                random_state=42
            )

            # mon = df_model_x['Monetary'].apply(lambda x: np.log(x))
            mon = df_model_x['Monetary']

            features = [[x, y, z] for x, (y, z) in zip(df_model_x['Recency'], zip(df_model_x['Frequency'], mon))]

            kmeans.fit(features)

            a = set(df_model_x['CustomerID'].unique()) - set(df_pred_x['CustomerID'].unique())
            a = list(a)
            d = df_model_x.loc[df_model_x['CustomerID'].isin(a)]

            label = kmeans.fit_predict(features)
            df_model_x['Group'] = label
            df_model_y['Group'] = label
            df_pred_x['Group'] = label
            df_pred_y['Group'] = label

            c = [int(sum(label == i)) for i in range(dg.get_group_count())]
            group = [int(np.argmax(c))]

            df_model_x = df_model_x.loc[df_model_x['Group'].isin(group)]
            df_model_y = df_model_y.loc[df_model_y['Group'].isin(group)]

            df_pred_x = df_pred_x.loc[df_pred_x['Group'].isin(group)]
            df_pred_y = df_pred_y.loc[df_pred_y['Group'].isin(group)]

        features = ['Recency', 'Frequency', 'Monetary', 'QuantityAvg', 'QuantityTotal', 'IsTrangle', 'IsEast', 'IsFu', 'IsWest']

        df_model_x = df_model_x.reset_index()[features]
        df_model_y = df_model_y.reset_index()[['Quantity']]

        df_pred_x = df_pred_x.reset_index()[features]
        df_pred_y = df_pred_y.reset_index()[['Quantity']]

        pr2.df_pred_x.append(df_pred_x)
        pr2.df_pred_y.append(df_pred_y)

        X_train, X_test, y_train, y_test = train_test_split(df_model_x, df_model_y, test_size=0.25, random_state=23)

        pr2.X_train.append(X_train)
        pr2.X_test.append(X_test)
        pr2.y_train.append(y_train)
        pr2.y_test.append(y_test)

    return pr2, months


def __train(pr2: PredictResult, months):
    for i, m in enumerate(months):
        X_train = pr2.X_train[i]
        y_train = pr2.y_train[i]

        linreg = XGBRegressor(
            random_state=20,
            n_jobs=-1,
        )

        model = linreg.fit(X_train, y_train)

        pr2.model.append(model)


def __train_combine(pr2: PredictResult, months, last_month):
    X_train_list = []
    y_train_list = []
    for i, m in enumerate(months):
        X_train = pr2.X_train[i]
        y_train = pr2.y_train[i]

        X_train_list.append(X_train)
        y_train_list.append(y_train)

        if m[1] == last_month:
            break

    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)

    linreg = XGBRegressor(
        random_state=20,
        n_jobs=-1,
    )

    model = linreg.fit(X_train, y_train)

    pr2.model = [model for _ in range(len(months))]


def __predict(pr2: PredictResult, months):
    for i, m in enumerate(months):
        model = pr2.model[i]
        X_train = pr2.X_train[i]
        y_train = pr2.y_train[i]
        X_test = pr2.X_test[i]
        y_test = pr2.y_test[i]
        df_pred_x = pr2.df_pred_x[i]
        df_pred_y = pr2.df_pred_y[i]

        train_pred_y = model.predict(X_train)
        test_pred_y = model.predict(X_test)

        pr2.train_pred_y.append(train_pred_y)
        pr2.test_pred_y.append(test_pred_y)

        pred_y = model.predict(df_pred_x)

        pr2.pred_y.append(pred_y)

        # get result
        mse_train = np.sqrt(mean_squared_error(y_train, train_pred_y))
        mae_train = mean_absolute_error(y_train, train_pred_y)
        mse_test = np.sqrt(mean_squared_error(y_test, test_pred_y))
        mae_test = mean_absolute_error(y_test, test_pred_y)
        mse_pred = np.sqrt(mean_squared_error(df_pred_y, pred_y))
        mae_pred = mean_absolute_error(df_pred_y, pred_y)

        pr2.mse_train.append(mse_train)
        pr2.mae_train.append(mae_train)
        pr2.mse_test.append(mse_test)
        pr2.mae_test.append(mae_test)
        pr2.mse_pred.append(mse_pred)
        pr2.mae_pred.append(mae_pred)

        ols_reg = sm.OLS(pred_y, df_pred_x)
        ols_reg = ols_reg.fit()
        rsquared = round(ols_reg.rsquared, 2)

        pr2.rsquared.append(rsquared)


def get_df_x(df, country_types, max_month, min_month):
    from datetime import datetime
    now = datetime.strptime(max_month, '%Y-%m')

    a = {t: np.mean for t in country_types}
    a['InvoiceMonth'] = np.max
    a['Quantity'] = ['mean', 'sum']
    a['USPrice'] = [lambda x: x.iloc[-1], lambda x: x.iloc[-2] if x.shape[0] > 1 else x.iloc[-1]]
    a['Monetary'] = np.sum
    a['InvoiceDate'] = [np.max, np.min]

    df_data = df.loc[(df['InvoiceMonthStr'] < max_month) & (df['InvoiceMonthStr'] >= min_month)]
    df_x = df_data.groupby('CustomerID').agg(a)
    df_x.columns = country_types + ['LastMonth', 'QuantityAvg', 'QuantityTotal', 'USPrice', 'USPriceLast', 'Monetary', 'LastPurchase', 'FirstPurchase', ]

    df_x = df_x.loc[df_x['LastPurchase'] != df_x['FirstPurchase']]

    df_x['TransCount'] = df.groupby('CustomerID').size()
    df_x['Frequency'] = df_x['TransCount'] * 30 / (df_x['LastPurchase'] - df_x['FirstPurchase']).dt.days
    df_x['Recency'] = (now - df_x['LastPurchase']).dt.days

    df_x = df_x.sort_values(by='CustomerID').reset_index()

    df_x['PriceRaise'] = (df_x['USPrice'] - df_x['USPriceLast']) / df_x['USPrice'] * 100

    # df_x = df_x.drop('LastMonth', 1)

    df_x = df_x.drop(['USPrice', 'USPriceLast', 'FirstPurchase', 'LastPurchase', 'TransCount', 'LastMonth', 'PriceRaise', ], 1)
    # df_x = df_x.drop([
    #     # 'IsTrangle',
    #     'IsTW',
    #     'IsSu',
    #     'IsWu',
    #     # 'IsCha',
    #     # 'IsChu',
    #     # 'IsNan',
    #     # 'IsOther',
    # ], 1)

    return df_x, df_data


def get_df_y(df, max_month, min_month, df_x):
    df_result = df.loc[df['InvoiceMonthStr'] == max_month].groupby('CustomerID').agg({
        'Quantity': np.sum,
    }).reset_index()

    df_q_dict = {c: i for c, i in zip(df_result['CustomerID'], df_result['Quantity'])}

    customerids = df_x['CustomerID'].unique()
    # display(len(customerids))
    customerids.sort()
    quantity = [df_q_dict[c] if c in df_q_dict else 0 for c in customerids]
    df_y = pd.DataFrame({
        'CustomerID': customerids,
        'Quantity': quantity,
    })

    return df_y


def generate_months(first_month, start_month, last_month, date_format, fix_first_month=False):
    months = []
    for i in range(month_dif(last_month, start_month) + 1):
        if fix_first_month:
            a = [
                add_month(first_month, 0 - 1),
                add_month(start_month, 0 - 1),
                add_month(first_month, i),
                add_month(start_month, i),
            ]
        else:
            a = [
                add_month(first_month, i - 1),
                add_month(start_month, i - 1),
                add_month(first_month, i),
                add_month(start_month, i),
            ]
        a = [m.strftime(date_format) for m in a]
        print(f'{a[0]}~{a[1]} -> {a[2]}~{a[3]}')
        months.append(a)
    return months


def month_dif(d1: datetime, d2: datetime):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def add_month(d1: datetime, month: int):
    if month == 0:
        return d1
    is_pos = month > 0
    month = abs(month)
    year = month // 12

    if year == 0:
        return d1 + relativedelta(months=month % 12) * (1 if is_pos else -1)
    else:
        return d1 + relativedelta(years=year, months=month % 12) * (1 if is_pos else -1)


def show_group_result(do_update):
    lang = st.session_state['page_lang']

    st.title(lang['analyze_reuslt'])

    group_result: list = st.session_state[mode_pred_next_month_cache]

    tabs = st.tabs([pg.data_group.get_displayname() for pg in group_result])

    for tab, (result) in zip(tabs, group_result):
        result: PredictResult
        name = result.data_group.get_displayname()

        with tab:
            month = result.pred_month

            st.write(f'{lang["average_customer"]}: ', f'{round(np.average([X_train.shape[0] for X_train in result.X_train]), 2)}')

            # correlation
            correlation_month = st.selectbox('Correlation', month, key=f'mode_pred_next_{result.data_group.get_id()}')
            model = result.model[month.index(correlation_month)]
            fig = plt.figure(figsize=(12, 4.5))
            plt.title(f'Correlation {correlation_month}')
            plt.barh(range(1, len(model.feature_importances_) + 1), model.feature_importances_, tick_label=result.X_train[month.index(correlation_month)].columns.tolist())
            st.pyplot(fig)

            # plot error
            key = f'{pn_single_error}_{name}'
            if not is_key_set(key) or do_update:
                st.session_state[key] = __draw_single_error(result, month)
            st.write('Error')
            st.image(st.session_state[key])

            # plot rsquared
            key = f'{pn_single_rsquare}_{name}'
            if not is_key_set(key) or do_update:
                st.session_state[key] = __draw_single_rsquared(result, month)
            st.write('Rsquared')
            st.image(st.session_state[key])

            # plot pred
            key = f'{pn_single_predict}_{name}'
            if not is_key_set(key) or do_update:
                st.session_state[key] = __draw_single_predict(result, month)
            st.write('Predict')
            st.image(st.session_state[key])

            # m = st.selectbox('Download result', result.pred_month, len(result.pred_month) - 1)
            # dt = result.pred_y[result.pred_month.index(m)]
            # print(dt)
            # st.download_button(
            #     'Download result as csv',
            #     dt,
            #     f'{result.data_group.get_displayname()}_result.csv',
            # )

    st.markdown('---')

    st.title(lang['cross_compare'])

    # plot error

    options = ['MSE train', 'MSE test', 'MSE pred', 'MAE train', 'MAE test', 'MAE pred', ]
    default_options = ['MSE test', 'MSE pred', 'MAE test', 'MAE pred', ]
    selected = st.multiselect('Error', options, default_options, key=pn_error_select, on_change=__on_cross_error_change)

    if not is_key_set(pn_error_changed) or st.session_state[pn_error_changed] or do_update:
        st.session_state[pn_error] = __draw_cross_error(group_result, month, selected)
    st.image(st.session_state[pn_error])
    st.session_state[pn_error_changed] = False

    # plot rsquared
    if not is_key_set(pn_rsquare) or do_update:
        st.session_state[pn_rsquare] = __draw_rsquare(group_result, month)
    st.write('Rsquared')
    st.image(st.session_state[pn_rsquare])


def __on_cross_error_change():
    st.session_state[pn_error_changed] = True


def __draw_single_error(result: PredictResult, month):
    fig = plt.figure(figsize=(12, 4.5))
    plt.title(f'{result.data_group.get_displayname()} error')

    plt.plot(result.mse_train, label='MSE train')
    plt.plot(result.mae_train, label='MAE train')
    plt.plot(result.mse_test, label='MSE test')
    plt.plot(result.mae_test, label='MAE test')
    plt.plot(result.mse_pred, label='MSE pred')
    plt.plot(result.mae_pred, label='MAE pred')

    plt.legend()
    plt.xticks(range(0, len(month)), month)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    return buf


def __draw_single_rsquared(result: PredictResult, month):
    fig = plt.figure(figsize=(12, 4.5))
    plt.title(f'{result.data_group.get_displayname()} rsquared')

    plt.plot(result.rsquared, label='rsquared')
    plt.xticks(range(0, len(month)), month)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    return buf


def __draw_single_predict(result: PredictResult, month):
    fig = plt.figure(figsize=(12, 4.5))
    plt.title(f'{result.data_group.get_displayname()} Predict')
    plt.plot([sum(r) for r in result.test_pred_y], label='Predict')
    plt.plot([sum(r['Quantity']) for r in result.y_test], label='Real')
    plt.xticks(range(0, len(month)), month)
    plt.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    return buf


def __draw_rsquare(group_result, month):
    fig = plt.figure(figsize=(12, 4.5))
    plt.title('rsquared')
    for result in group_result:
        result: PredictResult
        name = result.data_group.get_displayname()
        plt.plot(result.rsquared, label=f'{name} rsquared')
    plt.legend()
    plt.xticks(range(0, len(month)), month)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    return buf


def __draw_cross_error(group_result, month, selected):
    fig = plt.figure(figsize=(12, 4.5))
    plt.title(f'Error')

    for result in group_result:
        result: PredictResult
        name = result.data_group.get_displayname()

        if 'MSE train' in selected:
            plt.plot(result.mse_train, label=f'{name} MSE train')
        if 'MSE test' in selected:
            plt.plot(result.mse_test, label=f'{name} MSE test')
        if 'MSE pred' in selected:
            plt.plot(result.mse_pred, label=f'{name} MSE pred')
        if 'MAE train' in selected:
            plt.plot(result.mae_train, label=f'{name} MAE train')
        if 'MAE test' in selected:
            plt.plot(result.mae_test, label=f'{name} MAE test')
        if 'MAE pred' in selected:
            plt.plot(result.mae_pred, label=f'{name} MAE pred')

    plt.legend()
    plt.xticks(range(0, len(month)), month)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    return buf
