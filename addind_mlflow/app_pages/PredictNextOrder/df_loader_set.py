
import pandas as pd

path = '../data/pno/SC_ABC_RSBU.xlsx'

df = None


def get_selected_df(filepath=None):
    global df, path
    if filepath is not None:
        path = filepath
    if df is None:
        df = pd.read_excel(path)
        return df, True
    return df, False
