import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

# 读取个人信息
train_agg = pd.read_csv('../data/train/train_agg.csv',sep='\t')
test_agg = pd.read_csv('../data/test/test_agg.csv',sep='\t')

# 日志信息
train_log = pd.read_csv('../data/train/train_log.csv',sep='\t')
test_log = pd.read_csv('../data/test/test_log.csv',sep='\t')
# 用户唯一标识
train_flg = pd.read_csv('../data/train/train_flg.csv',sep='\t')
test_flg = pd.read_csv('../data/train/submit_sample.csv',sep='\t')
del test_flg['RST']

# 训练集合测试集切分EVT_LBL为三列
train_log['EVT_LBL_1'] = train_log['EVT_LBL'].map(lambda x:(str(x).split('-')[0]))
train_log['EVT_LBL_2'] = train_log['EVT_LBL'].map(lambda x: str(x).split('-')[1])
train_log['EVT_LBL_3'] = train_log['EVT_LBL'].map(lambda x: str(x).split('-')
[2])
test_log['EVT_LBL_1'] = test_log['EVT_LBL'].map(lambda x: str(x).split('-')[0])
test_log['EVT_LBL_2'] = test_log['EVT_LBL'].map(lambda x: str(x).split('-')[1])
test_log['EVT_LBL_3'] = test_log['EVT_LBL'].map(lambda x: str(x).split('-')[2])
# 训练集合测试集切分EVT_LBL1 EVT_LBL2 EVT_LBL交集
EVT_LBL_1=list(set(train_log['EVT_LBL_1']) & set(test_log['EVT_LBL_1']))
EVT_LBL_2=list(set(train_log['EVT_LBL_2']) & set(test_log['EVT_LBL_2']))
EVT_LBL_3=list(set(train_log['EVT_LBL_3']) & set(test_log['EVT_LBL_3']))

def User_log_Fun(feature_data):

    feature_data['hour'] = [int(str(i)[11:13].replace('-', '')) for i in feature_data['OCC_TIM']]
    feature_data['day'] = [int(str(i)[8:11].replace('-', '')) for i in feature_data['OCC_TIM']]

    # 统计次数
    def Count_feature(para_datas, keys):
        prefixs = 'l_'
        for key in keys:
            prefixs = key + '_'
        data = para_datas[keys]
        data['temp'] = 1
        data = data.groupby(keys).agg('sum').reset_index()
        data.rename(columns={'temp': prefixs + 'cnt'}, inplace=True)
        para_datas = pd.merge(para_datas, data, on=keys, how='left')
        para_datas=para_datas.drop_duplicates(keys)
        return para_datas

    User_log_user_id = list(set(feature_data['USRID']))
    User_log_user_id = pd.DataFrame(User_log_user_id, columns=['USRID'])
    User_log_user_id1 = pd.DataFrame(User_log_user_id, columns=['USRID'])

    # 统计USRID、EVT_LBL_1出现次数######################################################################################
    keys=['USRID','EVT_LBL_1']
    data = feature_data[keys].copy()
    data=Count_feature(data,keys)
    data= data[data['EVT_LBL_1'].map(lambda x:x in EVT_LBL_1)]
    data['EVT_LBL_1']=data['EVT_LBL_1'].map(lambda x:'EVT_LBL_1_'+str(x))
    data.set_index(keys,inplace=True)
    data=data.unstack(level=-1)
    data.reset_index(inplace=True)
    User_log_user_id=pd.merge(User_log_user_id, data, on=['USRID'], how='left')

    # 统计USRID、EVT_LBL_2出现次数
    keys = ['USRID', 'EVT_LBL_2']
    data = feature_data[keys].copy()
    data = Count_feature(data, keys)
    data = data[data['EVT_LBL_2'].map(lambda x: x in EVT_LBL_2)]
    data['EVT_LBL_2'] = data['EVT_LBL_2'].map(lambda x: 'EVT_LBL_2_' + str(x))
    data.set_index(keys, inplace=True)
    data = data.unstack(level=-1)
    data.reset_index(inplace=True)
    User_log_user_id = pd.merge(User_log_user_id, data, on=['USRID'], how='left')

    # 统计USRID、EVT_LBL_3出现次数
    keys = ['USRID', 'EVT_LBL_3']
    data = feature_data[keys].copy()
    data = Count_feature(data, keys)
    data = data[data['EVT_LBL_3'].map(lambda x: x in EVT_LBL_3)]
    data['EVT_LBL_3'] = data['EVT_LBL_3'].map(lambda x: 'EVT_LBL_3_' + str(x))
    data.set_index(keys, inplace=True)
    data = data.unstack(level=-1)
    data.reset_index(inplace=True)
    User_log_user_id = pd.merge(User_log_user_id, data, on=['USRID'], how='left')
    ######################################### 用户每天出现次数##########################################################
    for i in range(1, 32):
        data = feature_data[['USRID', 'day']].copy()
        data = data[data['day'] == i]
        data['user_log_USRID_per_day_last' + str(i) + '_cnt'] = 1
        del data['day']
        data = data.groupby(['USRID']).agg('sum').reset_index()
        User_log_user_id = pd.merge(User_log_user_id, data, on=['USRID'], how='left')

    # 用户出现方差、总和、平均数、最大值、最小值
    needs = []
    for col in User_log_user_id.columns.tolist():
        if 'user_log_USRID_per_day_last' in col:
            needs.append(col)
    User_log_user_id.fillna(0, inplace=True)
    User_log_user_id['user_log_USRID_per_day_cnt_var'] = User_log_user_id[needs].var(1)
    User_log_user_id['user_log_USRID_per_day_cnt_sum'] = User_log_user_id[needs].sum(1)
    User_log_user_id['user_log_USRID_per_day_cnt_avg'] = User_log_user_id[needs].mean(1)
    User_log_user_id['user_log_USRID_per_day_cnt_max'] = User_log_user_id[needs].max(1)
    User_log_user_id['user_log_USRID_per_day_cnt_min'] = User_log_user_id[needs].min(1)

    # ##差分计算每天出现次数######################################
    a = np.diff(User_log_user_id[needs])
    b = pd.DataFrame(a)
    b.columns = ['user_log_USRID_per_day_cnt_diff_1',
                 'user_log_USRID_per_day_cnt_diff_2', 'user_log_USRID_per_day_cnt_diff_3',
                 'user_log_USRID_per_day_cnt_diff_4', 'user_log_USRID_per_day_cnt_diff_5',
                 'user_log_USRID_per_day_cnt_diff_6', 'user_log_USRID_per_day_cnt_diff_7',
                 'user_log_USRID_per_day_cnt_diff_8', 'user_log_USRID_per_day_cnt_diff_9',
                 'user_log_USRID_per_day_cnt_diff_10', 'user_log_USRID_per_day_cnt_diff_11',
                 'user_log_USRID_per_day_cnt_diff_12', 'user_log_USRID_per_day_cnt_diff_13',
                 'user_log_USRID_per_day_cnt_diff_14', 'user_log_USRID_per_day_cnt_diff_15',
                 'user_log_USRID_per_day_cnt_diff_16','user_log_USRID_per_day_cnt_diff_17',
                 'user_log_USRID_per_day_cnt_diff_18', 'user_log_USRID_per_day_cnt_diff_19',
                 'user_log_USRID_per_day_cnt_diff_20', 'user_log_USRID_per_day_cnt_diff_21',
                 'user_log_USRID_per_day_cnt_diff_22', 'user_log_USRID_per_day_cnt_diff_23',
                 'user_log_USRID_per_day_cnt_diff_24', 'user_log_USRID_per_day_cnt_diff_25',
                 'user_log_USRID_per_day_cnt_diff_26', 'user_log_USRID_per_day_cnt_diff_27',
                 'user_log_USRID_per_day_cnt_diff_28', 'user_log_USRID_per_day_cnt_diff_29',
                 'user_log_USRID_per_day_cnt_diff_30']
    # 用户出现方差、总和、平均数、最大值、最小值
    needs = []
    for col in b.columns.tolist():
        if 'user_log_USRID_per_day_cnt_diff_' in col:
            needs.append(col)
    User_log_user_id['user_log_USRID_per_day_cnt_diff_var_cnt'] = b[needs].var(1)
    User_log_user_id['user_log_USRID_per_day_cnt_diff_sum_cnt'] = b[needs].sum(1)
    User_log_user_id['user_log_USRID_per_day_cnt_diff_avg_cnt'] = b[needs].mean(1)
    User_log_user_id['user_log_USRID_per_day_cnt_diff_max_cnt'] = b[needs].max(1)
    User_log_user_id['user_log_USRID_per_day_cnt_diff_min_cnt'] = b[needs].min(1)

    ## 时间间隔
    # 最近/远一次启动距离最近考察日的时间间隔
    data = feature_data[['USRID', 'day']].copy()
    data = data.groupby(['USRID'])['day'].agg({'user_log_USRID_min_day': np.min}).reset_index()
    User_log_user_id = pd.merge(User_log_user_id, data, on=['USRID'], how='left')
    User_log_user_id['furest_day_to_label']=User_log_user_id['user_log_USRID_min_day'].map(lambda x: 32 - x)


    data = feature_data[['USRID', 'day']].copy()
    data = data.groupby(['USRID'])['day'].agg({'user_log_USRID_max_day': np.max}).reset_index()
    User_log_user_id = pd.merge(User_log_user_id, data, on=['USRID'], how='left')
    User_log_user_id['near_day_to_label']=User_log_user_id['user_log_USRID_max_day'].map(lambda x: 32 - x)

    # # USRID TCH_TYP每个类型出现的次数#################################################################################
    for i in range(3):
        data = feature_data[['USRID', 'TCH_TYP']]
        data = data[data['TCH_TYP'] == i]
        data = data.groupby(['USRID']).agg('count').reset_index()
        data.rename(columns={'TCH_TYP': 'user_log_USRID_TCH_TYP_' + str(i) + '_cnt'}, inplace=True)
        User_log_user_id = pd.merge(User_log_user_id, data, on='USRID', how='left')

    # # USRID TCH_TYP每个类型每天出现的次数#############################################################################
    for ii in range(1,32):
        feature_datas=feature_data[['USRID', 'TCH_TYP', 'day']]
        feature_datas=feature_datas[feature_datas['day']==ii]
        del feature_datas['day']
        for i in range(3):
            data = feature_datas[['USRID', 'TCH_TYP']]
            data = data[data['TCH_TYP'] == i]
            data = data.groupby(['USRID']).agg('count').reset_index()
            data.rename(columns={'TCH_TYP': 'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_' + str(ii)+ ''}, inplace=True)
            User_log_user_id1 = pd.merge(User_log_user_id1, data, on='USRID', how='left')

    #TCH_TYP出现方差、总和、平均数、最大值、最小值
    User_log_user_id1.fillna(0, inplace=True)
    for i in range(3):
        needs = []
        for col in User_log_user_id1.columns.tolist():
            if 'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_' in col:
                needs.append(col)
        User_log_user_id.fillna(0, inplace=True)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_'+str(i)+'_cnt_var'] = User_log_user_id1[needs].var(1)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_'+str(i)+'_cnt_sum'] = User_log_user_id1[needs].sum(1)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_'+str(i)+'_cnt_mean'] = User_log_user_id1[needs].mean(1)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_'+str(i)+'_cnt_max'] = User_log_user_id1[needs].max(1)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_'+str(i)+'_cnt_min'] = User_log_user_id1[needs].min(1)

    # ##差分计算每天出现次数######################################
    for i in range(3):
        needs = []
        for col in User_log_user_id1.columns.tolist():
            if 'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_' in col:
                needs.append(col)
        a = np.diff(User_log_user_id1[needs])
        b = pd.DataFrame(a)
        b.columns = ['user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_1',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_2',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_3',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_4',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_5',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_6',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_7',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_8',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_9',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_10',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_11',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_12',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_13',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_14',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_15',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_16',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_17',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_18',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_19',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_20',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_21',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_22',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_23',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_24',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_25',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_26',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_27',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_28',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_29',
                     'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_30']
        # 用户出现方差、总和、平均数、最大值、最小值
        needs = []
        for col in b.columns.tolist():
            if 'user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_' in col:
                needs.append(col)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_var_cnt'] = b[needs].var(1)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_sum_cnt'] = b[needs].sum(1)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_avg_cnt'] = b[needs].mean(1)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_max_cnt'] = b[needs].max(1)
        User_log_user_id['user_log_USRID_TCH_TYP_per_day_' + str(i) + '_cnt_diff_min_cnt'] = b[needs].min(1)

    ## 时间间隔
    # 最近/远一次启动距离最近考察日的时间间隔
    for i in range(3):
        data = feature_data[['USRID', 'TCH_TYP','day']].copy()
        data=data[data['TCH_TYP']==i]
        del data['TCH_TYP']
        data = data.groupby(['USRID'])['day'].agg({'user_log_USRID_TCH_TYP_'+str(i)+'_min_day': np.min}).reset_index()
        User_log_user_id = pd.merge(User_log_user_id, data, on=['USRID'], how='left')
        User_log_user_id['furest_TCH_TYP_day_to_label'] = User_log_user_id['user_log_USRID_TCH_TYP_'+str(i)+'_min_day'].map(lambda x: 32 - x)

    for i in range(3):
        data = feature_data[['USRID', 'TCH_TYP','day']].copy()
        data=data[data['TCH_TYP']==i]
        del data['TCH_TYP']
        data = data.groupby(['USRID'])['day'].agg({'user_log_USRID_TCH_TYP_'+str(i)+'_max_day': np.max}).reset_index()
        User_log_user_id = pd.merge(User_log_user_id, data, on=['USRID'], how='left')
        User_log_user_id['near_TCH_TYP_day_to_label'] = User_log_user_id['user_log_USRID_TCH_TYP_'+str(i)+'_max_day'].map(lambda x: 32 - x)
    return User_log_user_id
def User_agg_Fun(feature_data):
    User_agg_user_id = list(set(feature_data['USRID']))
    User_agg_user_id = pd.DataFrame(User_agg_user_id, columns=['USRID'])

    # data = feature_data[['V2', 'V4', 'V5']].copy()
    # data['V2_V4'] = list(map(lambda x, y: str(x) + '_' + str(y), data['V2'], data['V4']))
    # data['V2_V4_V5'] = list(map(lambda x, y: str(x) + '_' + str(y), data['V2_V4'], data['V5']))
    # le = preprocessing.LabelEncoder()
    # le.fit(data['V2_V4_V5'])
    # data['V2_V4_V5_LabelEncoder'] = le.transform(data['V2_V4_V5'])
    # User_agg_user_id['V2_V4_V5_LabelEncoder'] = data['V2_V4_V5_LabelEncoder']
    return User_agg_user_id

def online_model(test,train):
    result = test[['USRID']]
    test.fillna(0, inplace=True)
    train.fillna(0, inplace=True)
    train_y = train.FLAG
    train_X = train.drop(['FLAG'], axis=1)

#    print(test)

    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_test = xgb.DMatrix(test)

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'scale_pos_weight': 1,
        'min_child_weight': 18,
    }
    num_rounds = 500# 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_train, 'verification')]

    # training model
    model = xgb.train(params, xgb_train, num_rounds)

    # 测试集
    preds = model.predict(xgb_test)
    test_pre_y = pd.DataFrame(preds)
    result['RST'] = test_pre_y

    result.to_csv('zhao.csv',index=False,sep='\t')


    return
if __name__ == '__main__':
    start_time = time.time()
    print('begin', start_time)

    # 训练集构造特征

    train_agg_feature_data = User_agg_Fun(train_agg)
    train_log_feature_data=User_log_Fun(train_log)
    train_data_label = pd.merge(train_flg, train_agg, on=['USRID'], how='left')


    # 测试集构造特征
    test_agg_feature_data = User_agg_Fun(test_agg)
    test_log_feature_data=User_log_Fun(test_log)
    test_data_label = pd.merge(test_flg, test_agg, on=['USRID'], how='left')


    # 合并训练集
    train_data_label=pd.merge(train_data_label,train_agg_feature_data,on=['USRID'], how='left')
    train_data_label = pd.merge(train_data_label, train_log_feature_data, on=['USRID'], how='left')
#    print(train_data_label)
    # 合并测试集
    test_data_label = pd.merge(test_data_label, test_agg_feature_data, on=['USRID'], how='left')
    test_data_label = pd.merge(test_data_label, test_log_feature_data, on=['USRID'], how='left')
#    print(test_data_label)

    # 线上预测
    online_model(test_data_label,train_data_label)

    print('end tiem', time.time() - start_time)

