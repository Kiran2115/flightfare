import os
import yaml
import pandas as pd
import argparse
from pkgutil import get_data
from get_data import read_params,get_data



def load_data(config_path):
    config = read_params(config_path)
    # print(config)
    load_data_path = config['load_data']['raw_dataset_csv']

    df = get_data(config_path)
    df.to_csv(load_data_path,index=False, sep=',')
    print('Data saved to raw')
    return load_data













if __name__ == "__main__":
      args = argparse.ArgumentParser()
      args.add_argument("--config",default="params.yaml")
      parsed_args = args.parse_args()
      load_data(config_path=parsed_args.config)