# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:15:08 2018

@author: CCL
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
import sys
import scipy as sp
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
OFF_LINE = True

def xgb_model(train_set_x,train_set_y,test_set_x):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 9,  # 2 3
              'silent':1
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    return predict

    
    
def log_tabel(data):
    EVT_LBL_len = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_len':len})
    EVT_LBL_set_len = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_set_len':lambda x:len(set(x))})
    TCH_TYP_set_len = data.groupby(by= ['USRID'], as_index = False)['TCH_TYP'].agg({'TCH_TYP_set_len':lambda x:len(set(x))})
    
    data['hour'] = data.OCC_TIM.map(lambda x:x.hour)
    data['day'] = data.OCC_TIM.map(lambda x:x.day)
    
    return EVT_LBL_len,EVT_LBL_set_len,TCH_TYP_set_len




    

    
    
if __name__ == '__main__':
    train_agg = pd.read_csv('../train/train_agg.csv',sep='\t')
    train_flg = pd.read_csv('../train/train_flg.csv',sep='\t')
    train_log = pd.read_csv('../train/train_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
    
    
    
    
    all_train = pd.merge(train_flg,train_agg,on=['USRID'],how='left')
    EVT_LBL_len,EVT_LBL_set_len,TCH_TYP_set_len = log_tabel(train_log)
    
    all_train = pd.merge(all_train,EVT_LBL_len,on=['USRID'],how='left')
    all_train = pd.merge(all_train,EVT_LBL_set_len,on=['USRID'],how='left')
    all_train = pd.merge(all_train,TCH_TYP_set_len,on=['USRID'],how='left')
    all_train.fillna(0,inplace=True)
    if OFF_LINE == True:    
        train_x = all_train.drop(['USRID', 'FLAG'], axis=1).values
        train_y = all_train['FLAG'].values
        auc_list = []
    
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        for train_index, test_index in skf.split(train_x, train_y):
            print('Train: %s | test: %s' % (train_index, test_index))
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
    
            pred_value = xgb_model(X_train, y_train, X_test)
            print(pred_value)
            print(y_test)
    
            pred_value = np.array(pred_value)
            pred_value = [ele + 1 for ele in pred_value]
    
            y_test = np.array(y_test)
            y_test = [ele + 1 for ele in y_test]
    
            fpr, tpr, thresholds = roc_curve(y_test, pred_value, pos_label=2)
            
            auc = metrics.auc(fpr, tpr)
            print('auc value:',auc)
            auc_list.append(auc)
    
        print('validate result:',np.mean(auc_list))
        sys.exit(32)
    
    
    test_agg = pd.read_csv('../test/test_agg.csv',sep='\t')
    test_log = pd.read_csv('../test/test_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
    EVT_LBL_len,EVT_LBL_set_len = log_tabel(test_log)
    test_set = pd.merge(test_agg,EVT_LBL_len,on=['USRID'],how='left')
    test_set = pd.merge(test_set,EVT_LBL_set_len,on=['USRID'],how='left')
    
    
    
    ###########################
    result_name = test_set[['USRID']]
    train_x = all_train.drop(['USRID', 'FLAG'], axis=1).values
    train_y = all_train['FLAG'].values
    test_x = test_set.drop(['USRID'], axis=1).values
    pred_result = xgb_model(train_x,train_y,test_x)
    result_name['RST'] = pred_result
    maxx = max(pred_result)
    minn = min(pred_result)
    result_name['RST'] = result_name['RST'].map(lambda x:(x-minn)/(maxx-minn))
    result_name.to_csv('test_result.csv',index=None,sep='\t')
    
    
