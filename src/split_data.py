import os
import yaml
import pandas as pd
import argparse
from pkgutil import get_data
from get_data import read_params,get_data
from sklearn.model_selection import train_test_split




def split_data(config_path):
    config = read_params(config_path)
    data_path = config['load_data']['raw_dataset_csv']
    data = pd.read_csv(data_path)
    test_ratio = config['split_data']['test_size']
    train,test = train_test_split(data, test_size=test_ratio)

    train_path = config['split_data']['train_path']
    test_path = config['split_data']['test_path']

    # df.to_csv(train_path,index=False, sep=',')
    train.to_csv(train_path)

    test.to_csv(test_path)



    print('train_test_split is done.')

    return split_data








if __name__ == "__main__":
      args = argparse.ArgumentParser()
      args.add_argument("--config",default="params.yaml")
      parsed_args = args.parse_args()
      split_data(config_path=parsed_args.config)