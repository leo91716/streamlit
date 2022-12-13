
import seaborn as sns
from app_pages.app_page import AppPage
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import base64
import pickle
from warnings import filterwarnings


from .reg_algos import Models
from .split import Split


filterwarnings(action='ignore')


class MLV2_V2_Page(AppPage):
    @staticmethod
    def _run_page():
        app()

    @staticmethod
    def get_name():
        return st.session_state['lang_config']['mlv2_V2']['name']


st.cache(suppress_st_warning=True)


def app():
    lang = st.session_state['lang_config']['mlv2_V2']

    # function for getting wider space on web page
    def _max_width_():
        max_width_str = f"max-width: 950px;"
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
    _max_width_()

    # seeting up title
    st.title(lang['title'])
    #st.markdown("<h1 style='text-align: center;'>?? UI ML - Regression ??</h1>", unsafe_allow_html=True)
    # st.markdown("<h1 style='text-align: center; color: white;'>?? UI ML - Regression ??</h1>", unsafe_allow_html=True)
    # user uploaded file
    st.sidebar.info(lang['side_bar_hint'])

    # asiking for custome file from local machine
    file_upload = st.sidebar.file_uploader(lang['upload_here'], type=['csv'], help=lang['upload_help'])
    # getting the name of sample data set name
    name = st.sidebar.selectbox(lang['select_sample_data'], ['None', 'california_housing', 'boston', 'diabetes'], help=lang['select_sample_help'])

    # smple file getting function
    def get_dataset(sample=True, custome=False):
        try:
            if sample:
                if name == 'boston':  # boston dataset
                    bos = datasets.load_boston()
                    df = pd.DataFrame(data=bos['data'], columns=bos['feature_names'])
                    df['Target'] = bos['target']
                    return df.sample(frac=0.5)  # returning 50% of total data for faster preprocessing
                elif name == 'california_housing':  # carlifornia housing dataset
                    df, y = datasets.fetch_california_housing(as_frame=True, return_X_y=True)
                    df['Target'] = y
                    return df.sample(frac=0.5)
                elif name == 'diabetes':  # diabetes dataset
                    df, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
                    df['Target'] = y  # adding target column
                    return df.sample(frac=0.5)
            if custome:
                df = pd.read_csv(file_upload)
                return df

        except Exception as ex:
            print(str(ex))

    # giving result by choosing dataset, custome or sample
    if file_upload is None:
        df = get_dataset()
    else:
        df = get_dataset(custome=True, sample=False)

    # showing the data.
    show_data = st.sidebar.checkbox(lang['show_data'], value=True)
    if show_data:
        st.write(lang['current_data'])
        st.write(df)

    # creating function to run algorithm
    def run_model(algo_c, preprocessor):
        try:
            algo = Models(X_train, y_train, X_test, y_test, preprocessor)
            if algo_c == 'Linear Regression':
                st.markdown(f'{lang["you_select"]} `Linear Regression`. [{lang["read_more"]}](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)')
                fit_intercept = col1.selectbox(lang['fit_intercept'], [True, False], help=lang['fit_intercept_help'])
                normalize = col2.selectbox(lang['normalize'], [False, True], help=lang['normalize_help'])
                n_jobs = col1.selectbox(lang['n_jobs'], [1, 4, -1], help=lang['n_jobs_help'])
                try:
                    score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model = algo.lin_reg(fit_intercept, normalize, n_jobs)
                    return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
                except Exception:
                    st.warning(lang['lr_warning'])

            elif algo_c == 'RandomForest Regressor':
                st.markdown(f'{lang["you_select"]} `RandomForest Regressor`. [{lang["read_more"]}](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)')
                n_estimators = col1.slider(lang['n_estimator'], 10, 500, 100, help=lang['n_estimator_help'])
                criterion = col2.selectbox(lang['criterion'], ['mse', 'mae'], help=lang['criterion_help'])
                max_depth = col1.slider(lang['max_depth'], 1, 50, help=lang['max_depth_help'])
                min_sample_split = col2.slider(lang['min_sample_split'], 2, 10, 3, help=lang['min_sample_split_help'])
                min_sample_leaf = col1.slider(lang['min_sample_leaf'], 1, 10, 1, help=lang['min_sample_leaf_help'])
                max_features = col2.selectbox(lang['max_features'], ['auto', 'sqrt', 'log2'], help=lang['max_features_help'])
                bootstrap = col1.selectbox(lang['bootstrap'], [True, False], help=lang['bootstrap_help'])
                max_samples = col2.slider(lang['max_sample'], 1, 10, 3, help=lang['max_sample_help'])
                try:
                    score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model = algo.rf_reg(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_sample_split=min_sample_split, min_sample_leaf=min_sample_leaf, max_features=max_features, bootstrap=bootstrap, max_samples=max_samples)
                    return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
                except Exception:
                    st.warning(lang['rr_warning'])

            elif algo_c == 'Support Vector Regression':
                st.markdown(f'{lang["you_select"]} `Support Vector Regression`. [{lang["read_more"]}](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)')
                # asking for hyperparameter
                kernel = col1.selectbox(lang['kernal'], ['rbf', 'linear', 'poly', 'sigmoid'], help=lang['kernal_help'])
                degree = col2.slider(lang['degree'], 1, 10, 3, help=lang['degree_help'])
                gamma = col1.selectbox(lang['gamma'], ['scale', 'auto'], help=lang['gamma_help'])
                coef0 = col2.number_input(lang['coef0'], help=lang['coef0_help'])
                C = col1.number_input(lang['c'], help=lang['c_help'])
                epsilon = col2.number_input(lang['epsilon'], help=lang['epsilon_help'])
                shrinking = col1.selectbox(lang['shrinking'], [True, False], help=lang['shrinking_help'])
                max_iter = col2.number_input(lang['max_iter'], help=lang['max_iter_help'])
                try:
                    score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model = algo.svr_reg(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, C=C, epsilon=epsilon, shrinking=shrinking, max_iter=max_iter)
                    return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
                except Exception:
                    st.warning(lang['svr_warning'])  # handeling bad choose of hyperparameter

            elif algo_c == 'KNeighbors Regressor':
                st.markdown(f'{lang["you_select"]} `KNeighbors Regressor`. [{lang["read_more"]}](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)')
                # asking for hyperparameter
                n_neighbors = col1.slider(lang['n_neighbors'], 1, 10, 5, help=lang['n_neighbors_help'])
                algorithm = col2.selectbox(lang['algorithm'], ['auto', 'ball_tree', 'kd_tree', 'brute'], help=lang['algorithm_help'])
                leaf_size = col1.slider(lang['leaf_size'], 1, 100, 30, help=lang['leaf_size_help'])
                p = col2.slider(lang['p'], 2, 10, 2, help=lang['p_help'])
                try:
                    score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model = algo.knn(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, p=p)
                    return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
                except Exception:
                    st.warning(lang['kn_warning'])
        except Exception as ex:
            print('run_model:' + str(ex))

    # starting data traning.
    try:
        if df is not None:
            df.dropna(axis=0, inplace=True)  # droping missing value if present
            col = list(df)
            col.insert(0, 'None')
            split = Split(df)
            # creating two column layout
            col1, col2 = st.columns(2)
            # creating alogorithm list
            algo_lst = ['Linear Regression', 'RandomForest Regressor', 'Support Vector Regression', 'KNeighbors Regressor']
            # taking target column to drop
            target = st.sidebar.selectbox(lang['select_col'], options=col, help=lang['select_col_help'])
            # algo slider
            algo_chose = st.sidebar.selectbox(lang['select_algo'], options=algo_lst, help=lang['select_algo_help'])
            # taking X and y from Split class
            try:
                X, y = split.X_and_y(target)
            except Exception:
                st.warning(lang['select_col_warning'])
            cat_col = X.select_dtypes(include=['object', 'category']).columns.tolist()
            num_col = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # numeric preprocess pipeline
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            # categorical feature transformer pipeline
            categorical_transformer = OneHotEncoder(drop='first', sparse=False)
            # now defining preprocessor
            if st.sidebar.checkbox(lang['apply_preprocessing'], value=False, help=lang['apply_preprocessing_help']):
                preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_col), ('cat', categorical_transformer, cat_col)])
            else:
                preprocessor = None  # if user not check then no preprocessing steps will apply.

            X_train, X_test, y_train, y_test = train_test_split(X, y)
            button = st.button(lang['train'], help=lang['train_help'])
            # data training process
            with st.spinner(lang['training']):
                score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model = run_model(algo_chose, preprocessor)

            if button:
                sns.set()
                col1, col2 = st.columns(2)
                col1.info(f'{lang["train_score"]}: ' + str(round(score_train, 3)))
                col2.info(f'{lang["test_score"]}: ' + str(round(score_test, 3)))
                col1.info(f'{lang["train_error"]}: ' + str(round(mae_train, 3)))
                col2.info(f'{lang["test_error"]}: ' + str(round(mae_test, 3)))
                # plotting traning and testing points
                st.write(f"<h3 style='text-align: center;'>{lang['train&test_plot']}</h3>", unsafe_allow_html=True)
                plt.figure(figsize=(15, 15))

                plt.subplot(2, 2, 1)
                plt.scatter(y_train, y_pred_train, c='b')  # plotting traning predction curve
                plt.xlabel(lang['training_label'])
                plt.ylabel(lang['training_predict'])
                plt.title(lang['training_plot'])
                plt.subplot(2, 2, 2)
                plt.scatter(y_test, y_pred_test, c='g')  # ploting testing prediction curve
                plt.xlabel(lang['test_label'])
                plt.ylabel(lang['test_predict'])
                plt.title(lang['test_plot'])
                # plotting residuals
                residual_train = (y_train - y_pred_train)
                residual_test = (y_test - y_pred_test)
                # plt.figure(figsize=(14,6))
                plt.subplot(2, 2, 3)
                sns.distplot(residual_train)
                plt.title(lang['training_residual'])
                plt.subplot(2, 2, 4)
                sns.distplot(residual_test)
                plt.title(lang['testing_residual'])

                st.set_option('deprecation.showPyplotGlobalUse', False)  # disableing plotting error
                st.pyplot()
                st.balloons()
                # saving the model and generate model download link
                output_model = pickle.dumps(model)
                b64 = base64.b64encode(output_model).decode()
                href = f'<a href="data:file/output_model;base64,{b64}" download="model.pkl">{lang["download_link"]}</a>'
                st.text(lang['can_download_link'])
                st.markdown(href, unsafe_allow_html=True)

        else:
            st.info(lang['upload_hint'])
    except Exception as ex:
        print('model_train ' + str(ex))
