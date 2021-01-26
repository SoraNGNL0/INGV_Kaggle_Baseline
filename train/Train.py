import lightgbm as lgbm
from sklearn.model_selection import KFold
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plots
import os
from pathlib import Path
from tqdm import tqdm
import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import yaml
# 仅使用二维特征+AdaBoost,并且不用五折交叉验证了试试

if __name__ == '__main__':
    DIR = "D:\\INGV_data"
    train = pd.read_csv(os.path.join(DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DIR, 'sample_submission.csv'))
    FE_path = ".\\FEdata\\model_trainFE1\\"
    train_set = pd.read_csv(os.path.join(FE_path, 'train_set.csv'))
    test_set = pd.read_csv(os.path.join(FE_path, 'test_set.csv'))
    log_path = ".\\log\\model_train17_5\\"
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    y = train_set['time_to_eruption']
    train_set = train_set.drop(['segment_id', 'time_to_eruption'], axis=1)
    test_set = test_set.drop(['segment_id', 'time_to_eruption'], axis=1)
    train_test_split_param = {
        'random_state': 42,
        'test_size': 0.2,
        'shuffle': True
    }
    AdaBoost_param = {
        'base_estimator': DecisionTreeRegressor(max_depth=20),
        'n_estimators': 150,
        'learning_rate': 0.0885,
        'loss': 'linear',
        'random_state': 66
    }
    X_train, X_val, y_train, y_val = train_test_split(train_set, y,
                                                      random_state=train_test_split_param['random_state'],
                                                      test_size=train_test_split_param['test_size'],
                                                      shuffle=train_test_split_param['shuffle'])

    add_val_mse = 0
    sub_preds = np.zeros(test_set.shape[0])
    model = AdaBoostRegressor(**AdaBoost_param)
    model.fit(X_train, y_train)
    val_mse = np.sqrt(mse(model.predict(X_val), y_val))
    print('AdaBoostRegressor validation rmse is ', val_mse)
    prediction = model.predict(test_set)

    # 将需要存储的东西全部存储至指定路径下
    time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    b = os.path.join(log_path, f'{time_now}')
    os.mkdir(b)
    test['time_to_eruption'] = prediction
    yamlpath = os.path.join(log_path, f'{time_now}', f'log.yaml')
    csv_file_name = os.path.join(log_path, f'{time_now}', f'submission.csv')
    test[['segment_id', 'time_to_eruption']].to_csv(csv_file_name, index=False)
    with open(yamlpath, "w", encoding="utf-8") as f:
        yaml.dump(str(train_test_split_param), f)
        yaml.dump(str(AdaBoost_param), f)
        yaml.dump(str(val_mse), f)
