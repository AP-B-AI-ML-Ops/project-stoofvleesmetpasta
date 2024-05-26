import datetime
import time
from dotenv import load_dotenv
import os
import joblib
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently import ColumnMapping

import psycopg
import pickle

load_dotenv()

# Define the numerical and categorical columns based on the student data
NUMERICAL = [
    'age',
    'absences',
    'G1',
    'G2',
    'G3'
]

CATEGORICAL = [
    'school',
    'sex',
    'adress',
    'Pstatus',
    'Medu',
    'Fedu',
    'Mjob',
    'Fjob',
    'reason',
    'guardian',
    'traveltime',
    'studytime',
    'failures',
    'schoolsup',
    'famsup',
    'paid',
    'activities',
    'nursery',
    'higher',
    'internet',
    'romantic',
    'famrel',
    'freetime',
    'goout',
    'health'
]

# Adjust the column mapping for the student data
COL_MAPPING = ColumnMapping(
    prediction='prediction',
    numerical_features=NUMERICAL,
    categorical_features=CATEGORICAL,
    target='alc'
)

# host, port, user, password
CONNECT_STRING = f'host={os.getenv("POSTGRES_HOST")} port={os.getenv("POSTGRES_PORT")} user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASSWORD")}'

def prep_db():
    create_table_query = """
    DROP TABLE IF EXISTS metrics;
    CREATE TABLE metrics(
        timestamp timestamp,
        prediction_drift float,
        num_drifted_columns integer,
        share_missing_values float
    );
    """

    with psycopg.connect(CONNECT_STRING, autocommit=True) as conn:
        # zoek naar database genaamd 'test' in de metadata van postgres
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test;")
        with psycopg.connect(f'{CONNECT_STRING} dbname=test') as conn:
            conn.execute(create_table_query)

def prep_data():
    # Load the prepared data
    with open('models/train.pkl', 'rb') as f_in:
        X_train, y_train = pickle.load(f_in)
    with open('models/val.pkl', 'rb') as f_in:
        X_val, y_val = pickle.load(f_in)
    with open('models/test.pkl', 'rb') as f_in:
        X_test, y_test = pickle.load(f_in)
    
    # Load the trained model
    with open('mlruns/3/327df86d9ca14817b89d566199cf66b8/artifacts/model/model.pkl', 'rb') as f_in:
        model = joblib.load(f_in)
    
    return X_train, y_train, X_test, y_test, model

def calculate_metrics(current_data, model, ref_data):
    current_data['prediction'] = model.predict(current_data.fillna(0))

    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    report.run(
        reference_data=ref_data,
        current_data=current_data,
        column_mapping=COL_MAPPING
    )

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_cols = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_vals = result['metrics'][2]['result']['current']['share_of_missing_values']

    return prediction_drift, num_drifted_cols, share_missing_vals

def save_metrics_to_db(cursor, date, prediction_drift, num_drifted_cols, share_missing_vals):
    cursor.execute("""
    INSERT INTO metrics(
        timestamp,
        prediction_drift,
        num_drifted_columns,
        share_missing_values
    )
    VALUES (%s, %s, %s, %s);
    """, (date, prediction_drift, num_drifted_cols, share_missing_vals))

def monitor():
    startDate = datetime.datetime(2022, 2, 1, 0, 0)
    endDate = datetime.datetime(2022, 2, 2, 0, 0)

    prep_db()

    ref_data, _, raw_data, _, model = prep_data()

    with psycopg.connect(f'{CONNECT_STRING} dbname=test') as conn:
        with conn.cursor() as cursor:
            # Simulate daily data checks
            for i in range(0, 27):
                current_data = raw_data[(raw_data.lpep_pickup_datetime >= startDate) &
                                        (raw_data.lpep_pickup_datetime < endDate)]
                
                prediction_drift, num_drifted_cols, share_missing_vals = calculate_metrics(current_data, model, ref_data)
                save_metrics_to_db(cursor, startDate, prediction_drift, num_drifted_cols, share_missing_vals)

                startDate += datetime.timedelta(1)
                endDate += datetime.timedelta(1)

                time.sleep(1)
                print("Data added")

if __name__ == '__main__':
    monitor()
