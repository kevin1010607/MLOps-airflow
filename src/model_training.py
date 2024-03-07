from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import os
import json
from typing import Tuple, List, Dict, Any

import mlflow


##########################################################################################
######################################## train ###########################################
##########################################################################################
# simple CatBoostRegressor
# multiple features and one target
# accept float or int data
# preprocess data -> split dataset -> train model -> save -> get report
##########################################################################################


def _FE(data: pd.DataFrame, target: str, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data.select_dtypes(include=['float', 'int']).shape != data.shape:
        raise Exception("data contain non float value")
    target = data[target]
    features = data[features]
    return target, features

def _DC(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()


def data_preprocessing(**kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Parameters
        data: unprocessed data
        target: target column name
        features: feature column names
    Returns
        target: target dataframe
        features: feature dataframe
    '''
    data = kwargs['data']
    target = kwargs['target']
    features = kwargs['features']

    try:
        data = _DC(data)
        target, features = _FE(data, target, features)
        return target, features
    except Exception as e:
        return e


def data_split(**kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    '''
    Parameters
        target: target dataframe
        features: feature dataframe
    Returns
        training_data: data used to train model
            {
                "feature": pd.DataFrame
                "target": pd.DataFrame
            }
        testing_data: data used to evaluate model
            {
                "feature": pd.DataFrame
                "target": pd.DataFrame 
    '''
    target, features = kwargs['task_instance'].xcom_pull(task_ids='data_preprocessing')
    try:
        training_features, testing_features, training_target, testing_target = train_test_split(features, target, random_state=42)
        training_data = {
            "feature": training_features,
            "target": training_target
        }
        testing_data = {
            "feature": testing_features,
            "target": testing_target,
        }
        return training_data, testing_data
    except Exception as e:
        return e


def train(**kwargs) -> Tuple[CatBoostRegressor, float]:
    '''
    Parameters
        training_data: data used to train model
            {
                "feature": pd.DataFrame
                "target": pd.DataFrame
            }
        testing_data: data used to evaluate model
            {
                "feature": pd.DataFrame
                "target": pd.DataFrame
            }
    Returns
        model: model
        score: model score
    '''
    training_data, testing_data = kwargs['task_instance'].xcom_pull(task_ids='data_split')

    try:
        estimator = CatBoostRegressor(random_state=42)
        model = estimator.fit(training_data["feature"], training_data["target"])
        score = model.score(testing_data["feature"], testing_data["target"])
        return model, score
    except Exception as e:
        return e

def save(**kwargs) -> None:
    '''
    Parameters
        model: model
        model_name: model file name
        path: model folder name
        target: target column name
        features: feature column names
        score: model r2 score
    Returns
    '''
    model, score = kwargs['task_instance'].xcom_pull(task_ids='train')
    path = kwargs['path']
    model_name = kwargs['model_name']
    target = kwargs['target']
    features = kwargs['features']

    try:
        result = {
            "target": target,
            "features": features,
            "score": score
        }
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, "result.json"), 'w') as f:
            json.dump(result, f)
        model.save_model(os.path.join(path, model_name))

        # mlflow
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("Test")
        with mlflow.start_run():
            # signature = mlflow.models.infer_signature(features, target)
            mlflow.log_metric("score", score)
            mlflow.set_tag("classifications", "simple CatBoostRegressor")
            mlflow.catboost.log_model(cb_model = model, artifact_path="test_model", registered_model_name="test_model")
    
    except Exception as e:
        return e


# Define the arguments for the DAG
dag_id = 'model_training'
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 2, 21),
}
schedule = None
description = 'model training'

with DAG(
    dag_id, 
    default_args=default_args, 
    schedule=schedule, 
    description=description, 
) as dag:
    # Task 1: data preprocessing
    data = os.path.join(os.path.dirname(__file__), 'input', 'train', 'data.csv')
    data = pd.read_csv(data, dtype=np.float64)
    target_name = 'target'
    features_name = [f'col{i}' for i in range(1, 30)]
    task1 = PythonOperator(
        task_id='data_preprocessing', 
        python_callable=data_preprocessing, 
        op_kwargs={
            'data': data, 
            'target': target_name, 
            'features': features_name
        }, 
        provide_context=True, 
        dag=dag, 
    )

    # Task 2: data split
    task2 = PythonOperator(
        task_id='data_split', 
        python_callable=data_split, 
        provide_context=True, 
        dag=dag, 
    )

    # Task 3: train
    task3 = PythonOperator(
        task_id='train', 
        python_callable=train, 
        provide_context=True, 
        dag=dag, 
    )

    # Task 4: save
    path = os.path.join(os.path.dirname(__file__), 'output', 'train')
    model_name = 'model.pkl'
    task4 = PythonOperator(
        task_id='save', 
        python_callable=save, 
        op_kwargs={
            'path': path, 
            'model_name': model_name, 
            'target': target_name, 
            'features': features_name
        }, 
        provide_context=True, 
        dag=dag, 
    )

    # Define task dependencies
    task1 >> task2 >> task3 >> task4
