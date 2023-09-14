import pandas as pd
import numpy as np

remove = open('stage01-remove.txt').read().splitlines()
na_zero = open('stage01-na_zero.txt').read().splitlines()
leak = open('stage01-leak.txt').read().splitlines()
remove = set(remove)
remove.update(leak)

def read_data(subset):
    frame = pd.read_csv('ELSO_data_3.csv', index_col='RunID')
    
    frame[na_zero] = frame[na_zero].fillna(0)
    columns = [column for column in frame.columns if column not in remove]
    frame = frame[columns]
    
    frame['Transplant_Yes'] = frame['Transplant'] == 'Yes'
    frame['Sex_Male'] = frame['Sex'] == 'Male'
    frame['PreECLSArrest_Yes'] = frame['PreECLSArrest'] == 'Yes'
    frame.drop(['PreECLSArrest', 'Transplant', 'Sex'], axis=1, inplace=True)
    frame = frame.dropna()
    if subset:
        frame = frame[subset]
    return frame

def discretize(frame, bins):
    names = list(frame.columns)
    columns = []
    for name in names:
        if name in bins:
            columns.append(pd.get_dummies(pd.cut(frame[name], bins = bins[name]), prefix = name))
        else:
            columns.append(frame[name])
    return pd.concat(columns, axis=1)
