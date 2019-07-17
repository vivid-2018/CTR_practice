# coding = utf-8
import numpy as np
import pandas as pd
from args import *
from DIN import DIN


def helper(x):
    if x == '':
        return 0
    return len(x.strip('|').split('|'))


def process_data(df):
    for s in features:
        df[s] = df[s].apply(str)
    df_x, df_y = df[features].values, df['label'].values.reshape((-1,1))
    hist_len = df[history_col].apply(helper).values
    hist_item = []
    for line in df[history_col]:
        ids = []
        for val in line.strip('|').split('|'):
            ids.append(item2index[val])
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        else:
            ids = ids[-max_len:]
        hist_item.append(ids)
    hist_item = np.array(hist_item)
    return df_x, df_y, hist_item, hist_len




