# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:46:48 2018

@author: FNo0
"""

import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

def load_data():
    # source train
    train_agg = pd.read_csv(r'../data/train/train_agg.csv',delimiter = '\t')
    train_flg = pd.read_csv(r'../data/train/train_flg.csv',delimiter = '\t')
    train_log = pd.read_csv(r'../data/train/train_log.csv',delimiter = '\t')
    # source test
    test_agg = pd.read_csv(r'../data/test/test_agg.csv',delimiter = '\t')
    test_log = pd.read_csv(r'../data/test/test_log.csv',delimiter = '\t')
    # 预处理
    train_log['EVT_LBL_1'] = train_log['EVT_LBL'].map(lambda x : x.split('-')[0])
    train_log['EVT_LBL_2'] = train_log['EVT_LBL'].map(lambda x : x.split('-')[1])
    train_log['EVT_LBL_3'] = train_log['EVT_LBL'].map(lambda x : x.split('-')[2])
    train_log.drop(['EVT_LBL'],axis = 1,inplace = True)
    train_log['OCC_TIM'] = pd.to_datetime(train_log['OCC_TIM'])
    train_log['OCC_DAY'] = train_log['OCC_TIM'].map(lambda x : x.day)
    train_log['OCC_HOUR'] = train_log['OCC_TIM'].map(lambda x : x.hour)
    test_log['EVT_LBL_1'] = test_log['EVT_LBL'].map(lambda x : x.split('-')[0])
    test_log['EVT_LBL_2'] = test_log['EVT_LBL'].map(lambda x : x.split('-')[1])
    test_log['EVT_LBL_3'] = test_log['EVT_LBL'].map(lambda x : x.split('-')[2])
    test_log.drop(['EVT_LBL'],axis = 1,inplace = True)
    test_log['OCC_TIM'] = pd.to_datetime(test_log['OCC_TIM'])
    test_log['OCC_DAY'] = test_log['OCC_TIM'].map(lambda x : x.day)
    test_log['OCC_HOUR'] = train_log['OCC_TIM'].map(lambda x : x.hour)
    ## 联合编码疑似离散值的特征列：V2,V4
    train_agg['V2V4'] = list(map(lambda x,y : str(x) + '_' + str(y),train_agg['V2'],train_agg['V4']))
    test_agg['V2V4'] = list(map(lambda x,y : str(x) + '_' + str(y),test_agg['V2'],test_agg['V4']))
    le = preprocessing.LabelEncoder()
    le.fit(train_agg['V2V4'])
    train_agg['V2V4'] = le.transform(train_agg['V2V4'])
    test_agg['V2V4'] = le.transform(test_agg['V2V4'])
    ## 联合编码疑似离散值的特征列：V2,V5
    train_agg['V2V5'] = list(map(lambda x,y : str(x) + '_' + str(y),train_agg['V2'],train_agg['V5']))
    test_agg['V2V5'] = list(map(lambda x,y : str(x) + '_' + str(y),test_agg['V2'],test_agg['V5']))
    le = preprocessing.LabelEncoder()
    le.fit(train_agg['V2V5'])
    train_agg['V2V5'] = le.transform(train_agg['V2V5'])
    test_agg['V2V5'] = le.transform(test_agg['V2V5'])
    ## 联合编码疑似离散值的特征列：V4,V5
    train_agg['V4V5'] = list(map(lambda x,y : str(x) + '_' + str(y),train_agg['V4'],train_agg['V5']))
    test_agg['V4V5'] = list(map(lambda x,y : str(x) + '_' + str(y),test_agg['V4'],test_agg['V5']))
    le = preprocessing.LabelEncoder()
    le.fit(train_agg['V4V5'])
    train_agg['V4V5'] = le.transform(train_agg['V4V5'])
    test_agg['V4V5'] = le.transform(test_agg['V4V5'])
    ## 联合编码疑似离散值的特征列：V2,V4,V5
    train_agg['V2V4V5'] = list(map(lambda x,y,z : str(x) + '_' + str(y) + '_' + str(z),train_agg['V2'],train_agg['V4'],train_agg['V5']))
    test_agg['V2V4V5'] = list(map(lambda x,y,z : str(x) + '_' + str(y) + '_' + str(z),test_agg['V2'],test_agg['V4'],test_agg['V5']))
    le = preprocessing.LabelEncoder()
    le.fit(train_agg['V2V4V5'])
    train_agg['V2V4V5'] = le.transform(train_agg['V2V4V5'])
    test_agg['V2V4V5'] = le.transform(test_agg['V2V4V5'])
    # 返回
    return train_agg,train_flg,train_log,test_agg,test_log

def get_log_feat(dataset,EVT_LBL_1,EVT_LBL_2,EVT_LBL_3):
    data = dataset.copy()
    data['cnt'] = 1
    # 返回的特征
    feature = pd.DataFrame(columns = ['USRID'])
    
    ## 每天点击
    # pivot + unstack
    pivot = pd.pivot_table(data,index = ['USRID','OCC_DAY'],values = 'cnt',aggfunc = len)
    pivot = pivot.unstack(level = -1)
    pivot.fillna(0,downcast = 'infer',inplace = True)
    # 特征
    feat = pd.DataFrame()
    feat['USRID'] = pivot.index
    feat.index = pivot.index
    ## 统计特征
    # 总和
    feat['USRID_click_OCC_DAY_sum'] = pivot.sum(1)
    # 每一天的特征
    dates = list(set(data['OCC_DAY'].tolist()))
    for i in dates:
        for j in pivot.columns.tolist():
            if i == j[-1]:
                feat['USRID_click_OCC_DAY_' + str(i)] = pivot[j]
    ## 添加进特征
    feature = pd.merge(feature,feat,on = ['USRID'],how = 'outer')
    print('USRID_OCC_DAY特征提取完毕!')
    
    ## 每种TCH_TYP点击
    # pivot + unstack
    pivot = pd.pivot_table(data,index = ['USRID','TCH_TYP'],values = 'cnt',aggfunc = len)
    pivot = pivot.unstack(level = -1)
    pivot.fillna(0,downcast = 'infer',inplace = True)
    # 特征
    feat = pd.DataFrame()
    feat['USRID'] = pivot.index
    feat.index = pivot.index
    # 每种TCH_TYP的特征
    tchtyps = list(set(data['TCH_TYP'].tolist()))
    for i in tchtyps:
        for j in pivot.columns.tolist():
            if i == j[-1]:
                feat['USRID_click_TCH_TYP_' + str(i)] = pivot[j]
    ## 添加进特征
    feature = pd.merge(feature,feat,on = ['USRID'],how = 'outer')
    print('USRID_TCH_TYP特征提取完毕!')
    
    ## 每种EVT_LBL_1
    # pivot + unstack
    pivot = pd.pivot_table(data,index = ['USRID','EVT_LBL_1'],values = 'cnt',aggfunc = len)
    pivot = pivot.unstack(level = -1)
    pivot.fillna(0,downcast = 'infer',inplace = True)
    # 特征
    feat = pd.DataFrame()
    feat['USRID'] = pivot.index
    feat.index = pivot.index
    # 每种EVT_LBL_1的特征
    for i in EVT_LBL_1:
        for j in pivot.columns.tolist():
            if i == j[-1]:
                feat['USRID_click_EVT_LBL_1_' + str(i)] = pivot[j]
    ## 添加进特征
    feature = pd.merge(feature,feat,on = ['USRID'],how = 'outer')
    print('USRID_EVT_LBL_1特征提取完毕!')
    
    ## 每种EVT_LBL_2
    # pivot + unstack
    pivot = pd.pivot_table(data,index = ['USRID','EVT_LBL_2'],values = 'cnt',aggfunc = len)
    pivot = pivot.unstack(level = -1)
    pivot.fillna(0,downcast = 'infer',inplace = True)
    # 特征
    feat = pd.DataFrame()
    feat['USRID'] = pivot.index
    feat.index = pivot.index
    # 每种EVT_LBL_2的特征
    for i in EVT_LBL_2:
        for j in pivot.columns.tolist():
            if i == j[-1]:
                feat['USRID_click_EVT_LBL_2_' + str(i)] = pivot[j]
    ## 添加进特征
    feature = pd.merge(feature,feat,on = ['USRID'],how = 'outer')
    print('USRID_EVT_LBL_2特征提取完毕!')
    
    ## 每种EVT_LBL_3
    # pivot + unstack
    pivot = pd.pivot_table(data,index = ['USRID','EVT_LBL_3'],values = 'cnt',aggfunc = len)
    pivot = pivot.unstack(level = -1)
    pivot.fillna(0,downcast = 'infer',inplace = True)
    # 特征
    feat = pd.DataFrame()
    feat['USRID'] = pivot.index
    feat.index = pivot.index
    # 每种EVT_LBL_3的特征
    for i in EVT_LBL_3:
        for j in pivot.columns.tolist():
            if i == j[-1]:
                feat['USRID_click_EVT_LBL_3_' + str(i)] = pivot[j]
    ## 添加进特征
    feature = pd.merge(feature,feat,on = ['USRID'],how = 'outer')
    print('USRID_EVT_LBL_3特征提取完毕!')
    
    ## 返回
    return feature

def create_data(agg,log,EVT_LBL_1,EVT_LBL_2,EVT_LBL_3):
    data = agg.copy()
    # USRID特征
    feat = get_log_feat(log,EVT_LBL_1,EVT_LBL_2,EVT_LBL_3)
    data = pd.merge(data,feat,on = ['USRID'],how = 'left')
    data.fillna(0,downcast = 'infer',inplace = True)
    # 返回
    return data

def get_dataset():
    # 原始数据
    train_agg,train_flg,train_log,test_agg,test_log = load_data()
    ## 训练集和测试集都有的EVT_LBL
    EVT_LBL_1 = list(set(train_log['EVT_LBL_1']) & set(test_log['EVT_LBL_1']))
    EVT_LBL_2 = list(set(train_log['EVT_LBL_2']) & set(test_log['EVT_LBL_2']))
    EVT_LBL_3 = list(set(train_log['EVT_LBL_3']) & set(test_log['EVT_LBL_3']))
    EVT_LBL_1.sort()
    EVT_LBL_2.sort()
    EVT_LBL_3.sort()
    EVT_LBL_1.reverse()
    EVT_LBL_2.reverse()
    EVT_LBL_3.reverse()
    ## 构造训练集、测试集
    # 训练集
    print('构造训练集:')
    train = create_data(train_agg,train_log,EVT_LBL_1,EVT_LBL_2,EVT_LBL_3)
    train = pd.merge(train,train_flg,on = 'USRID',how = 'right')
    print('训练集构造完成!')
    print()
    # 测试集
    print('构造测试集:')
    test = create_data(test_agg,test_log,EVT_LBL_1,EVT_LBL_2,EVT_LBL_3)
    print('测试集构造完成!')
    # 返回
    return train,test

def model_xgb(tr,te):
    train = tr.copy()
    test = te.copy()
    
    train_y = train['FLAG'].values
    train_x = train.drop(['USRID','FLAG'],axis=1).values
    test_x = test.drop(['USRID'],axis=1).values
 
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eval_metric' : 'error',
              'eta': 0.03,
              'max_depth': 6,  # 4 3
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18  # 2 3
              }
    # 训练
    print('开始训练!')
    bst = xgb.train(params, dtrain, num_boost_round=300)
    # 预测
    print('开始预测!')
    predict = bst.predict(dtest)
    test_xy = test[['USRID']]
    test_xy['RST'] = predict
    test_xy.sort_values(['RST'],ascending = False,inplace = True)
    return test_xy

if __name__ == '__main__':
    tr,te = get_dataset()
    result = model_xgb(tr,te)
    result.to_csv('gao.csv',index=None,sep='\t')

