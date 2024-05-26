import os
import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from prefect import task, flow
import pickle

@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    
@task
def load_data():
    df = pd.read_csv('data/student-mat.csv')
    return df

@task
def clean_data(df):
    df["alc"] = df['Walc'] + df ['Dalc']
    return df.dropna()

@task
def remove_target(df):
    df = df.drop(columns=["alc", "Dalc", "Walc"])
    return df

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
def prep_flow(data_path, dest_path):
    # Load parquet files
    df_train = load_data()
    df_val = load_data()
    df_test = load_data()

    #preprocess data
    df_train = encode_data(clean_data(df_train))
    df_val = encode_data(clean_data(df_val))
    df_test = encode_data(clean_data(df_test))


    # Extract the target
    target = 'alc'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Create X
    X_train = remove_target(df_train)
    X_test = remove_target(df_test)
    X_val = remove_target(df_val)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
