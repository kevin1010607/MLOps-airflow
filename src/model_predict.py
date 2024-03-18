from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import os
import json
from typing import Tuple, List, Dict, Any

import mlflow

##########################################################################################
####################################### predict ##########################################
##########################################################################################
# predict by given model path and feature
# preprocess data -> predict -> save -> get report
##########################################################################################


def _FE(data: pd.DataFrame) -> pd.DataFrame:
    if data.select_dtypes(include=['float', 'int']).shape != data.shape:
        raise Exception("data contain non float value")
    return data

def _DC(data: pd.DataFrame) -> pd.DataFrame:
    data.dropna()
    return data


def data_preprocessing(**kwargs) -> pd.DataFrame:
    '''
    Parameters
        data: unprocessed feature dataframe
    Returns
        data: processed feature dataframe
    '''
    data = kwargs['data']

    try:
        data = _FE(data)
        data = _DC(data)
        return data
    except Exception as e:
        return e

def predict(**kwargs) -> pd.DataFrame:
    '''
    Parameters
        model_path: model location
        data: predict feature dataframe
    Returns
        result: predictions
    '''
    data = kwargs['task_instance'].xcom_pull(task_ids='data_preprocessing')
    model_path = kwargs['model_path']

    try:
        # mlflow
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("Test")
        client = mlflow.MlflowClient()
        latest_version = client.search_model_versions("name='test_model'")[0].version
        model_uri = f"models:/test_model/{latest_version}"
        loaded_model = mlflow.catboost.load_model(model_uri)
        predictions = loaded_model.predict(data)

        # estimator = CatBoostRegressor()
        # model = estimator.load_model(model_path)
        # predictions = model.predict(data)
        # return pd.DataFrame(predictions)
    except Exception as e:
        return e

def save(**kwargs) -> None:
    '''
    Parameters
        path: model folder name
        result: contain predict result
            format:
                {"predict": predict result}
    Returns
    '''
    result = kwargs['task_instance'].xcom_pull(task_ids='predict')
    path = kwargs['path']

    try:
        res = {
            "predict": {
                    "0": result.to_dict()[0]
                }
        }
        with open(os.path.join(path, "predict.json"), 'w') as f:
            json.dump(res, f)
    except Exception as e:
        return e


# Define the arguments for the DAG
dag_id = 'model_predict'
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 2, 21),
}
schedule = None
description = 'model predict'

with DAG(
    dag_id, 
    default_args=default_args, 
    schedule=schedule, 
    description=description, 
) as dag:
    # Task 1: data preprocessing
    data = os.path.join(os.path.dirname(__file__), 'input', 'predict', 'data.csv')
    data = pd.read_csv(data, dtype=np.float64)
    targets_name = ['target']
    features_name = [f'col{i}' for i in range(1, 30)]
    targets = data[targets_name]
    features = data[features_name]
    data = features
    task1 = PythonOperator(
        task_id='data_preprocessing', 
        python_callable=data_preprocessing, 
        op_kwargs={
            'data': data
        }, 
        provide_context=True, 
        dag=dag, 
    )

    # Task 2: predict
    model_name = 'model.pkl'
    model_path = os.path.join(os.path.dirname(__file__), 'input', 'predict', model_name)
    task2 = PythonOperator(
        task_id='predict', 
        python_callable=predict, 
        op_kwargs={
            'model_path': model_path
        }, 
        provide_context=True, 
        dag=dag, 
    )

    # Task 3: save
    path = os.path.join(os.path.dirname(__file__), 'output', 'predict')
    task3 = PythonOperator(
        task_id='save', 
        python_callable=save, 
        op_kwargs={
            'path': path, 
        }, 
        provide_context=True, 
        dag=dag, 
    )

    # Define task dependencies
    task1 >> task2 >> task3
