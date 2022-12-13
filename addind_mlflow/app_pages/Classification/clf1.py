
from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import base64
from warnings import filterwarnings

from app_pages.app_page import AppPage
from .classification_to_df import classification_report_to_dataframe
from .my_plot_confusion_matrix import my_confusion_matrix
from .split import Split
from .clasi_algos import Models


filterwarnings(action='ignore')


class CLF1Page(AppPage):
    @staticmethod
    def _run_page():
        app()

    @staticmethod
    def get_name():
        return st.session_state['lang_config']['clf1']['name']


def app():
    lang = st.session_state['lang_config']['clf1']

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

    # writing title of the app
    st.markdown(f"<h1 style='text-align: center;'>{lang['title']}</h1>", unsafe_allow_html=True)

    # asking for file
    st.sidebar.info(lang['side_bar_hint'])

    # file uploader
    file_upload = st.sidebar.file_uploader(label=lang['upload_here'], type=['csv'], help=lang['upload_help'])

    # slection of sample dataset
    sample_datasets_name = st.sidebar.selectbox(lang['select_sample_data'], options=('None', 'Iris Flowers', 'Wine', 'Breast Cancer'), help=lang['select_sample_help'])

    # function for getting the dataset

    def get_dataset(sample=True, custome=False):
        try:
            if sample:
                if sample_datasets_name == 'Iris Flowers':
                    df, y = datasets.load_iris(return_X_y=True, as_frame=True)
                    df['Flowers Name (Target Column)'] = y  # adding targeget column to dataframe
                    return df
                elif sample_datasets_name == 'Wine':
                    df, y = datasets.load_wine(return_X_y=True, as_frame=True)
                    df['Profile (Target Column)'] = y
                    return df
                elif sample_datasets_name == 'Breast Cancer':
                    df, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
                    df['Class (Target Column)'] = y
                    return df

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
    if st.sidebar.checkbox(lang['show_data'], True):
        st.write(lang['current_data'])
        st.write(df)

    col1, col2 = st.columns(2)

    def run_model(algo_c, preprocessor):
        try:
            algo = Models(X_train, y_train, X_test, y_test, preprocessor)
            if algo_c == 'Logistic Regression':
                st.write(f'{lang["you_select"]} `Logistic Regression`.[{lang["read_more"]}](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)')
                penalty = col1.selectbox(lang['penalty'], options=['l2', 'l1', 'none', 'elasticnet'], help=lang['penalty_help'])
                dual = col2.selectbox(lang['dual'], options=[False, True], help=lang['dual_help'])
                C = col1.slider(lang['c'], 1.0, 5.0, 0.25, help=lang['c_help'])
                solver = col2.selectbox(lang['solver'], options=['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'], help=lang['solver_help'])
                multi_class = col1.selectbox(lang['multi_class'], options=['auto', 'ovr', 'multinomial'], help=lang['multi_class_help'])
                f1, classi_rep, y_pred, model = algo.log_reg(penalty=penalty, dual=dual, C=C, solver=solver, multi_class=multi_class)  # model instance
                return f1, classi_rep, y_pred, model

            elif algo_c == "RandomForest Classifier":
                st.write(f'{lang["you_select"]} `RandomForest Classifier`. [{lang["read_more"]}](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)')
                n_estimators = col1.slider(lang['n_estimators'], 50, 500, 100, help=lang['n_estimators_help'])
                max_depth = col2.slider(lang['max_depth'], 1, 10, 3, help=lang['max_depth_help'])
                criterion = col1.selectbox(lang['criterion'], options=['gini', 'entropy'], help=lang['criterion_help'])
                min_samples_split = col2.slider(lang['min_samples_split'], 2, 10, 3, help=lang['min_samples_split_help'])
                min_samples_leaf = col1.slider(lang['min_samples_leaf'], 1, 10, 1, help=lang['min_samples_leaf_help'])
                max_features = col2.selectbox(lang['max_features'], options=['auto', 'sqrt', 'log2'], help=lang['max_features_help'])
                bootstrap = col1.selectbox(lang['bootstrap'], options=[True, False], help=lang['bootstrap_help'])
                max_samples = col2.slider(lang['max_samples'], 1, 10, 3, help=lang['max_samples_help'])
                f1, classi_rep, y_pred, model = algo.rnd_frst(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap, max_samples=max_samples)
                return f1, classi_rep, y_pred, model

            elif algo_c == 'Support Vector Classifier':
                st.write(f'{lang["you_select"]} `Support Vector Classifier`. [{lang["read_more"]}](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)')
                C = col1.slider(lang['SVC_C'], 1.0, 10.0, 0.25, help=lang['SVC_C_help'])
                kernel = col2.selectbox(lang['kernel'], options=['rbf', 'linear', 'ploy', 'sigmoid'], help=lang['kernel_help'])
                degree = col1.slider(lang['degree'], 1, 10, 1, help=lang['degree_help'])
                coefo = col2.slider(lang['coefo'], 0.0, 5.0, 0.25, help=lang['coefo_help'])
                f1, classi_rep, y_pred, model = algo.svc(C=C, kernel=kernel, degree=degree, coef0=coefo)
                return f1, classi_rep, y_pred, model

            elif algo_c == 'KNeighbors Classifier':
                st.write(f'{lang["you_select"]} `KNeighbors Classifier`.[{lang["read_more"]}](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)')
                n_neighbours = col1.slider(lang['n_neighbours'], 1, 10, 1, help=lang['n_neighbours_help'])
                weights = col2.selectbox(lang['weights'], ['uniform', 'distance'], help=lang['weights_help'])
                algorithm = col1.selectbox(lang['algorithm'], options=['auto', 'ball_tree', 'kd_tree', 'brute'], help=lang['algorithm_help'])
                leaf_size = col2.slider(lang['leaf_size'], 10, 100, 10, help=lang['leaf_size_help'])
                p = col1.selectbox(lang['p'], options=[2, 1], help=lang['p_help'])
                f1, classi_rep, y_pred, model = algo.knn(n_neighbors=n_neighbours, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
                return f1, classi_rep, y_pred, model
        except Exception as ex:
            print(lang['run_model_error'], str(ex))

    if df is not None:
        try:
            col = list(df)
            # col = col.insert(0, 'None')
            split = Split(df)
            algo_lst = ['Logistic Regression', 'RandomForest Classifier', 'Support Vector Classifier', 'KNeighbors Classifier']
            target = st.sidebar.selectbox(lang['select_col'], options=col)
            algo_chose = st.sidebar.selectbox(lang['select_algo'], options=algo_lst, help=lang['select_algo_help'])
            X, y = split.X_and_y(target)
            # getting categorical columns and numerical columns list
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
            with st.spinner(lang['training']):
                f1, classi_rep, y_pred, model = run_model(algo_chose, preprocessor)
        except Exception as e:
            st.warning(lang['train_error'])
        button = st.button(lang['train'], help=lang['train_help'])

        if button:
            st.text(lang['f1_score'] + str(round(f1, 3)))
            classi_df = classification_report_to_dataframe(classi_rep)
            st.text(lang['classification_report'])
            st.dataframe(classi_df)
            with st.spinner(lang['creat_conf_matrix']):
                my_confusion_matrix(y_test, y_pred, figsize=(5, 5), text_size=7)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            # download the model
            output_model = pickle.dumps(model)
            b64 = base64.b64encode(output_model).decode()
            href = f'<a href="data:file/output_model;base64,{b64}" download="model.pkl">{lang["download_link"]}</a>'
            st.text(lang['can_download_link'])
            st.markdown(href, unsafe_allow_html=True)
            st.balloons()

    else:
        st.warning(lang['upload_hint'])
