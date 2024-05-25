import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from prefect import flow

@flow
def train_model():
    mlflow.start_run()
    
    train = pd.read_csv('data/train.csv')
    val = pd.read_csv('data/val.csv')
    
    X_train = train.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
    y_train = train['target_column']
    
    X_val = val.drop('target_column', axis=1)
    y_val = val['target_column']
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    mlflow.end_run()

