import os

from kaggle.api.kaggle_api_extended import KaggleApi
from prefect import flow, task


@task
def authenticate_kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api


@task(retries=4, retry_delay_seconds=2)
def download_dataset(api, dataset: str):
    api.dataset_download_files(dataset, path="./data", unzip=True)


@flow
def collect_flow(update=False):
    os.makedirs("./data", exist_ok=True)
    api = authenticate_kaggle_api()

    if (not os.path.exists("./data/raw/consumption.csv")) or update:
        download_dataset(api, "uciml/student-alcohol-consumption")
