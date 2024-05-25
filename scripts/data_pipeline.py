from prefect import task, flow
from scripts.data_preparation import data_cleaning
from scripts.model_training import train_model



@flow
def pipeline_flow():
    data_cleaning_flow = data_cleaning()
    train_and_log_dlow = train_model()

