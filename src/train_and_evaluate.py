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



            #print('X and Y are created')
            if list(config['estimators'].keys())[0] == 'ElasticNet':
                    model = ElasticNet(alpha=config['estimators']['ElasticNet']['params']['alpha'],
                                        l1_ratio = config['estimators']['ElasticNet']['params']['l1_ratio'], random_state = config['base']['random_state'] )

                    model.fit(X,Y)
                    filename = f"{config['model_dir']}/{list(config['estimators'].keys())[0]}_model.sav"
                    joblib.dump(model, filename)

                    #params_json = model.get_params()
                    params_json = {
                        'alpha': config['estimators']['ElasticNet']['params']['alpha'],
                        'l1_ratio': config['estimators']['ElasticNet']['params']['l1_ratio']
                    }
                    with open(config['reports']['params'], "w") as outfile:
                                      json.dump(params_json, outfile)

                    test_pred = model.predict(test_X)
                    mae = mean_absolute_error(test_pred,test_Y)
                    mse = mean_squared_error(test_pred,test_Y)
                    rmse = np.sqrt(mse)
                    r2score = r2_score(test_Y,test_pred)

                    scores_dict = {'mean_squared_error':mse,'mean_absolute_error':mae,'root_mean_squared_error':rmse,'r2_score':r2score}

                    with open(config['reports']['scores'], "w") as outfile2:
                                      json.dump(scores_dict, outfile2)


                    print('Model saved in the models folder')
                    return train_model

            else:
                   print("You haven't configured those models") 
                   return train_model

                          




if __name__ == "__main__":
      args = argparse.ArgumentParser()
      args.add_argument("--config",default="params.yaml")
      parsed_args = args.parse_args()
      train_model(config_path=parsed_args.config)