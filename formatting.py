import pandas as pd

columns = ['name', 'f', 'p', 'r', 'text']


def gen_serie(name, rouge, text):
    return pd.Series([
        name,
        rouge[0]['rouge-l']['f'],
        rouge[0]['rouge-l']['p'],
        rouge[0]['rouge-l']['r'],
        text],
        index=columns)
