from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import os
import json
from sklearn import metrics
import scipy.stats as st
from typing import Tuple, List, Dict, Any

import mlflow


##########################################################################################
####################################### evaluate #########################################
##########################################################################################
# evaluate by given predict target and real target
# evaluate model -> save -> get report
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
        return pd.DataFrame(predictions)
    except Exception as e:
        return e

def evaluate(**kwargs) -> Dict:
    '''
    Parameters
        predictions: predict target
        targets: real target
    Returns
        regression scores
            format:
                {
                    "r2_score": r2_score,
                    "mse_score": mse_score,
                    "pos_max_err": pos_max_err,
                    "neg_max_err": neg_max_err,
                    "interval": interval
                }
    '''
    predictions = kwargs['task_instance'].xcom_pull(task_ids='predict')
    targets = kwargs['targets']

    try:
        # evaluate
        confidence = 99
        r2_score = metrics.r2_score(targets, predictions)
        mse_score = metrics.mean_squared_error(targets, predictions)
        pos_max_err = np.max(np.array(predictions)-np.array(targets))
        neg_max_err = np.min(np.array(predictions)-np.array(targets))

        error = np.square((np.array(targets) - np.array(predictions)))
        if len(error)>2:
            CL = np.abs(st.norm.ppf((1-confidence/100)/2))
            interval = np.sqrt(1/(len(error)-2))*CL
        else:
            interval = None

        res = {
            "r2_score": r2_score,
            "mse_score": mse_score,
            "pos_max_err": pos_max_err,
            "neg_max_err": neg_max_err,
            "interval": interval
        }

        # mlflow
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("Test")
        client = mlflow.MlflowClient()
        latest_model_run_id = client.search_model_versions("name='test_model'")[0].run_id 
        for key in res:
            client.log_metric(latest_model_run_id, key, res[key])

        return res
    except Exception as e:
        return e

def save(**kwargs) -> None:
    '''
    Parameters
        path: model folder name
        result: contain evalute score
            format:
                {
                    "r2_score": r2_score,
                    "mse_score": mse_score,
                    "pos_max_err": pos_max_err,
                    "neg_max_err": neg_max_err,
                    "interval": interval
                }
    Returns
    '''
    result = kwargs['task_instance'].xcom_pull(task_ids='evaluate')
    path = kwargs['path']

    try:
        with open(os.path.join(path, "evalute.json"), 'w') as f:
            json.dump(result, f)
    except Exception as e:
        return e


# Define the arguments for the DAG
dag_id = 'model_evaluate'
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 2, 21),
}
schedule = None
description = 'model evaluate'

with DAG(
    dag_id, 
    default_args=default_args, 
    schedule=schedule, 
    description=description, 
) as dag:
    # Task 1: data preprocessing
    data = os.path.join(os.path.dirname(__file__), 'input', 'evaluate', 'data.csv')
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
    model_path = os.path.join(os.path.dirname(__file__), 'input', 'evaluate', model_name)
    task2 = PythonOperator(
        task_id='predict', 
        python_callable=predict, 
        op_kwargs={
            'model_path': model_path
        }, 
        provide_context=True, 
        dag=dag, 
    )

    # Task 3: evaluate
    task3 = PythonOperator(
        task_id='evaluate', 
        python_callable=evaluate, 
        op_kwargs={
            'targets': targets
        }, 
        provide_context=True, 
        dag=dag, 
    )

    # Task 4: save
    path = os.path.join(os.path.dirname(__file__), 'output', 'evaluate')
    task4 = PythonOperator(
        task_id='save', 
        python_callable=save, 
        op_kwargs={
            'path': path, 
        }, 
        provide_context=True, 
        dag=dag, 
    )

    # Define task dependencies
    task1 >> task2 >> task3 >> task4
