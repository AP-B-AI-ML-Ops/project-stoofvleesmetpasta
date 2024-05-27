import mlflow
from load.collect import collect_flow
from load.prep import prep_flow
from prefect import flow
from train.hpo import hpo_flow
from train.register import register_flow
from train.train import train_flow

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
REG_EXPERIMENT_NAME = "random-forest-best-models"


@flow
def main_flow():
    print("start main flow")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    collect_flow("./data/")
    prep_flow("./data/")

    train_flow("./data/")
    hpo_flow("./data/", 5, HPO_EXPERIMENT_NAME)
    register_flow("./data/", 5, REG_EXPERIMENT_NAME, HPO_EXPERIMENT_NAME)


if __name__ == "__main__":
    main_flow()
