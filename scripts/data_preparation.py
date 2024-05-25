import os
import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from prefect import task, flow

@task
def load_data():
    df = pd.read_csv('data/raw/.csv')
    return df

@task
def clean_data(df):
    # Assuming there are no significant cleaning steps required
    return df.dropna()

@task
def encode_data(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

@task
def split_data(df):
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return train, val, test
@flow
def data_cleaning():
    download_and_extract_data()
    df = load_data()
    df = clean_data(df)
    df = encode_data(df)
    train, val, test = split_data(df)

    train.to_csv('data/train.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    test.to_csv('data/test.csv', index=False)
