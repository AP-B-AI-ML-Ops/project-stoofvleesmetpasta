import datetime
import time
from dotenv import load_dotenv
import os
import joblib
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently import ColumnMapping
import psycopg
import pickle

# Load environment variables from .env file
load_dotenv()

# Define numerical and categorical columns for student data
NUMERICAL = [
    'age', 'absences', 'G1', 'G2', 'G3', 'Medu', 'Fedu', 'traveltime', 'studytime',
    'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'
]

CATEGORICAL = [
    'school_MS', 'sex_M', 'address_U', 'famsize_LE3', 'Pstatus_T', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_health', 'Fjob_other', 'Fjob_services', 
    'reason_home','reason_other','reason_reputation','guardian_mother','guardian_other', 'guardian_other', 'schoolsup_yes', 'famsup_yes', 'paid_yes', 'activities_yes', 'nursery_yes', 'higher_yes',
    'internet_yes', 'romantic_yes'
]

# Configure column mapping for the dataset
COL_MAPPING = ColumnMapping(
    prediction='prediction',
    numerical_features=NUMERICAL,
    categorical_features=CATEGORICAL,
    target='alc'
)

# Database connection string
CONNECT_STRING = f"host={os.getenv('POSTGRES_HOST')} port={os.getenv('POSTGRES_PORT')} user={os.getenv('POSTGRES_USER')} password={os.getenv('POSTGRES_PASSWORD')}"

def prep_db():
    """Prepare the database by creating the necessary tables."""
    create_table_query = """
    DROP TABLE IF EXISTS metrics;
    CREATE TABLE metrics (
        timestamp TIMESTAMP,
        prediction_drift FLOAT,
        num_drifted_columns INTEGER,
        share_missing_values FLOAT
    );
    """
    with psycopg.connect(CONNECT_STRING, autocommit=True) as conn:
        # Check if the 'test' database exists and create it if not
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test;")
        with psycopg.connect(f"{CONNECT_STRING} dbname=test") as conn:
            conn.execute(create_table_query)

def prep_data():
    """Load the prepared data and the trained model."""
    with open('models/val.pkl', 'rb') as f_in:
        X_val, y_val = pickle.load(f_in)
    with open('models/test.pkl', 'rb') as f_in:
        X_test, y_test = pickle.load(f_in)
    with open('mlruns/3/327df86d9ca14817b89d566199cf66b8/artifacts/model/model.pkl', 'rb') as f_in:
        model = joblib.load(f_in)

    X_val.describe()
    
    # Make predictions for the reference data
    ref_data = X_val.copy()
    ref_data['prediction'] = model.predict(ref_data.fillna(0))
    
    return ref_data, y_val, X_test, y_test, model

def calculate_metrics(current_data, model, ref_data):
    """Calculate metrics for the given data using the Evidently library."""
    current_data = current_data.copy()  # Avoid modifying the original DataFrame

    # Ensure the 'prediction' column is not present in the current data
    if 'prediction' in current_data.columns:
        current_data.drop(columns=['prediction'], inplace=True)

    # Add the 'prediction' column to the current data
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
    """Save the calculated metrics to the database."""
    cursor.execute("""
    INSERT INTO metrics (
        timestamp,
        prediction_drift,
        num_drifted_columns,
        share_missing_values
    )
    VALUES (%s, %s, %s, %s);
    """, (date, prediction_drift, num_drifted_cols, share_missing_vals))

def monitor():
    """Main function to monitor and log metrics daily."""
    startDate = datetime.datetime(2022, 2, 1)
    prep_db()

    ref_data, _, raw_data, _, model = prep_data()

    with psycopg.connect(f"{CONNECT_STRING} dbname=test") as conn:
        with conn.cursor() as cursor:
            for i in range(0, 27):
                if len(raw_data) > i:
                    current_data = raw_data.iloc[[i]]  # Simulate daily data

                    prediction_drift, num_drifted_cols, share_missing_vals = calculate_metrics(current_data, model, ref_data)
                    save_metrics_to_db(cursor, startDate, prediction_drift, num_drifted_cols, share_missing_vals)

                    startDate += datetime.timedelta(days=1)

                    time.sleep(1)
                    print("Data added")

if __name__ == '__main__':
    monitor()
