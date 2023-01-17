import os
import yaml
import pandas as pd
import argparse
import joblib
import numpy as np
from pkgutil import get_data
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from get_data import get_data, read_params
import mlflow 
from urllib.parse import urlparse


def train_model(config_path):
            config = read_params(config_path)
            train = pd.read_csv(config['split_data']['train_path'])
            #print(train.head())
            X = train.drop(config['base']['target_col'],axis=1)
            Y = train[config['base']['target_col']]

            test = pd.read_csv(config['split_data']['test_path'])
            #print(train.head())
            test_X = test.drop(config['base']['target_col'],axis=1)
            test_Y = test[config['base']['target_col']]



            mlflow_config = config['mlflow_config']

            remote_server_uri = mlflow_config['remote_server_uri']

            mlflow.set_tracking_uri(remote_server_uri)

            mlflow.set_experiment(mlflow_config['experiment_name'])

            with mlflow.start_run(run_name=mlflow_config['run_name']) as mlops_run:

                       

                lr = ElasticNet(alpha=config['estimators']['ElasticNet']['params']['alpha'],

                                        l1_ratio = config['estimators']['ElasticNet']['params']['l1_ratio'], random_state = config['base']['random_state'] )



                lr.fit(X,Y)

                test_pred = lr.predict(test_X)



                mae = mean_absolute_error(test_pred,test_Y)

                mse = mean_squared_error(test_pred,test_Y)

                r2score = r2_score(test_pred,test_Y)

                rmse = np.sqrt(mse)



                mlflow.log_param("alpha",config['estimators']['ElasticNet']['params']['alpha'])

                mlflow.log_param("l1_ratio",config['estimators']['ElasticNet']['params']['l1_ratio'])




                mlflow.log_metric("rmse",rmse)

                mlflow.log_metric("mae",mae)

                mlflow.log_metric("r2",r2score)



                tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme



                if tracking_url_type_store != "file":

                         mlflow.sklearn.log_model(lr,"model",registered_model_name = mlflow_config['registered_model_name'])



                else:

                         mlflow.sklearn.load_model(lr,"model")
                          




if __name__ == "__main__":
      args = argparse.ArgumentParser()
      args.add_argument("--config",default="params.yaml")
      parsed_args = args.parse_args()
      train_model(config_path=parsed_args.config)