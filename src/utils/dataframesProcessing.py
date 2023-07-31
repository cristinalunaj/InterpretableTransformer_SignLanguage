import pandas as pd

def process_padding(df, last_padd=-2, new_padd=0):
    dfNewPad = df.replace(to_replace=last_padd, value=new_padd)
    return dfNewPad


def cleanNaNrows(df):
    df = df.dropna(axis=1, how='all')
    return df


def get_condition(dataframe, paddVal):
    # create series with all values set to true
    condition = pd.Series([True] * len(dataframe))
    for col in dataframe.columns:
        condition &= dataframe[col] == paddVal
    return condition


def cleanRepeated_rows(df, paddVal=-2):
    condition = get_condition(df, paddVal)
    df = df[~condition]
    df = df.reset_index(drop=True)
    print("to do")
    return df

