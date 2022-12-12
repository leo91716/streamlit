

from io import BytesIO
import pandas as pd
import subprocess
import os


def load_excel(file: BytesIO, sheet_name=None):
    xlsx_file = '.temp.xlsx'
    csv_file = '.temp.csv'

    with open(xlsx_file, 'wb') as f:
        f.write(file.getbuffer())

    if type(sheet_name) is list:
        names = sheet_name
    else:
        names = [sheet_name]

    df_list = []
    for name in names:
        if name is not None:
            call = ["python", "./util/xlsx2csv.py", '-n', name, xlsx_file, csv_file]
        else:
            call = ["python", "./util/xlsx2csv.py", xlsx_file, csv_file]

        subprocess.call(call)

        df_list.append(pd.read_csv(csv_file))

    if len(df_list) == 1:
        df = df_list[0]
    else:
        df = pd.concat(df_list)

    os.remove(csv_file)
    os.remove(xlsx_file)

    return df
