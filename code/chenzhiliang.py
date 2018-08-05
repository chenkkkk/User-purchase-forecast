# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:18:19 2018

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
import time
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from xgboost import plot_importance
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 

columns_rank_list = ['user_EVT3_3169','user_EVT3_2067','user_EVT3_3744','user_EVT2_1261','user_EVT3_271','V30','V23','user_EVT2_1798','user_EVT3_3928','user_EVT3_3840','user_EVT3_4334','user_EVT3_3785','user_day_16',\
'user_EVT3_3211','user_EVT2_43','user_EVT3_3923','user_EVT3_4374','user_EVT2_20','user_EVT3_3159','user_day_10','user_day_5','user_EVT3_1688','user_EVT1_181','user_day_2','user_EVT3_3922','user_EVT3_4282','user_EVT3_4383',\
'user_EVT3_291','user_EVT3_3155','user_EVT3_3862','user_day_11','user_EVT3_3137','user_EVT3_3851','user_EVT3_1706','user_EVT2_555','user_EVT3_568','user_EVT3_2035','user_EVT3_3136','user_EVT3_4326','user_EVT2_1797','user_EVT2_224',\
'user_EVT3_3857','user_EVT3_2068','user_EVT2_1047','user_EVT3_4317','user_EVT2_1043','user_EVT3_3754','user_EVT3_3145','user_EVT3_3662','user_EVT2_1914','user_EVT2_233','user_EVT3_2071','user_EVT2_1270','user_EVT3_3868','user_EVT1_139',\
'user_EVT3_4314','user_EVT3_565','user_EVT2_2141','user_EVT2_2157','user_EVT2_2158','user_EVT3_3931','user_EVT3_2206','user_EVT3_3804','user_EVT2_1591','user_EVT2_221','user_EVT2_227','user_EVT3_3807','user_EVT3_3827','user_EVT3_3856',\
'user_EVT3_3873','user_EVT3_3670','user_EVT3_3787','user_EVT1_259','V7','user_EVT3_1525','user_EVT3_3835','user_EVT3_3836','user_EVT3_3765','user_EVT3_2205','user_EVT3_3829','user_EVT3_4338','user_EVT3_4380','user_EVT2_1483','V25',\
'user_EVT2_2162','user_EVT2_578','user_EVT3_2637','user_EVT3_1715','user_EVT3_3927','user_EVT3_1199','user_EVT3_273','user_EVT3_3801','user_EVT3_4351','user_EVT3_4353','user_EVT3_3871','user_EVT3_4393','user_EVT3_1703','user_EVT3_3813',\
'user_EVT2_1852','user_EVT2_702','user_EVT2_2143','user_EVT1_102','user_EVT3_3182','user_EVT2_1233','user_EVT3_2037','user_EVT3_274','user_EVT2_1846','user_EVT3_2644','user_EVT3_3730','user_EVT3_3929','user_EVT3_3847','user_EVT3_566',\
'user_EVT3_4367','user_EVT2_1843','V29','user_EVT3_3149','user_EVT3_4373','user_EVT3_3618','user_EVT3_1724','user_EVT3_3615','user_EVT3_561','user_EVT3_3494','user_EVT3_3869','user_EVT3_1719','user_EVT3_1197','user_EVT3_3865','user_EVT1_359',\
'user_EVT2_1234','user_EVT2_2164','user_EVT1_10','user_EVT2_1593','V3','user_EVT1_162','user_EVT2_2140','gap_log_day_min','user_day_7','user_EVT3_2049','user_EVT3_563','user_EVT3_2642','user_EVT2_1828','V9','user_EVT3_4360','user_EVT2_21',\
'user_EVT3_3820','user_EVT3_4366','user_EVT2_1044','user_EVT3_4304','user_day_21','user_EVT3_3749','user_EVT3_2077','user_EVT3_2204','user_EVT3_3841','user_EVT2_1040','user_EVT3_4330','user_EVT3_564','user_EVT2_2154','user_EVT3_2045','user_EVT1_460',\
'user_EVT3_1722','user_EVT3_17','user_EVT3_2062','user_EVT3_4280','user_EVT3_1682','user_EVT3_268','user_EVT3_4378','user_EVT2_231','user_EVT3_2056','user_day_22','user_EVT2_2136','user_EVT3_3127','user_EVT3_3860','user_EVT3_3170','user_EVT3_2643',\
'user_EVT3_3800','user_EVT3_3614','user_EVT3_3158','user_EVT2_1482','user_EVT2_561','user_EVT2_15','user_EVT3_3728','user_EVT2_704','user_EVT3_1681','user_EVT3_912','user_EVT2_1911','user_day_4','user_EVT3_3142','user_day_24','user_EVT3_3133','user_EVT3_3864',\
'user_EVT2_2138','user_EVT3_1710','user_EVT3_3623','user_EVT3_4318','user_EVT2_229','user_EVT3_1704','user_EVT3_22','user_EVT3_3129','user_EVT3_4288','user_EVT3_3737','user_EVT3_3805','user_EVT3_3621','V6','user_EVT2_2148','user_EVT2_1910','user_EVT3_4391',\
'user_EVT3_3247','user_EVT1_604','V10','user_EVT3_3144','user_EVT3_3739','user_EVT3_3179','user_EVT2_2156','user_EVT3_4392','user_EVT3_4328','user_EVT3_3830','V11','user_EVT3_2004','user_EVT3_3202','user_day_28','user_EVT3_3121','user_EVT2_574','user_EVT2_1916',\
'user_EVT3_1677','user_EVT3_3861','user_EVT3_4289','user_EVT3_3736','user_EVT3_3828','user_EVT3_3814','user_EVT3_3842','user_EVT3_1679','user_EVT2_924','user_EVT3_4396','user_EVT3_3852','user_EVT3_1702','user_EVT3_1716','user_EVT3_1727','user_EVT2_1048',\
'user_EVT3_3126','user_day_14','user_EVT2_1481','user_EVT2_1848','user_EVT3_4311','user_EVT3_1689','user_EVT1_163','user_day_18','user_EVT2_395','user_EVT3_3138','user_EVT2_575','user_EVT2_1260','user_EVT3_569','user_EVT3_3770','user_EVT3_1693','user_EVT3_3846',\
'user_EVT3_3769','user_EVT3_3775','user_EVT3_4359','V28','user_EVT2_16','user_EVT3_3134','user_EVT2_1351','user_EVT3_3788','user_EVT3_1523','user_EVT3_4381','user_EVT2_2137','user_EVT3_3619','user_EVT3_3132','user_EVT3_3767','user_EVT3_4364','user_EVT3_2638',\
'user_EVT3_1690','user_EVT3_2050','user_EVT2_2146','user_EVT2_1847','user_EVT3_560','user_EVT3_4299','user_EVT3_4365','user_day_30','user_EVT3_2031','user_EVT3_4344','user_EVT3_1687','user_day_15','user_EVT2_2163','user_EVT3_3786','user_EVT3_4291','user_EVT2_1913',\
'user_day_6','user_day_31','user_EVT3_2639','user_EVT1_38','user_EVT3_1692','user_EVT3_3930','user_EVT3_1198','user_EVT2_1269','user_EVT3_3156','user_EVT2_1826','user_EVT3_2061','user_EVT3_18','user_EVT2_1841','user_EVT3_3866','user_EVT2_1042','user_EVT3_3748',\
'user_EVT3_2640','user_EVT2_1831','user_EVT2_1854','user_EVT1_520','user_EVT2_1851','user_day_27','user_EVT3_2079','user_EVT3_3789','user_EVT3_4322','user_EVT3_4294','user_EVT2_577','user_EVT3_2072','user_EVT3_3731','user_EVT3_3243','user_EVT2_1235',\
'user_EVT3_3755','user_EVT3_3773','user_EVT3_3757','user_EVT3_4387','user_EVT1_0','user_EVT2_576','V1','user_EVT3_3747','user_EVT3_4349','user_EVT2_1049','user_EVT3_3751','user_EVT3_3838','user_EVT3_3141','user_EVT2_1352','user_EVT3_4302','user_EVT3_3832',\
'user_EVT2_553','user_EVT3_1200','user_EVT3_4329','user_EVT3_1694','user_EVT3_570','user_EVT3_893','user_EVT3_4295','user_EVT2_2135','user_EVT3_117','user_EVT3_572','user_EVT3_3753','user_EVT2_1842','user_EVT3_267','user_EVT3_3776','user_EVT3_20','user_EVT3_2003',\
'user_EVT3_4301','user_day_25','user_EVT3_913','user_EVT3_3771','gap_log_day_max','user_EVT3_3199','user_EVT3_2046','user_EVT3_3128','user_EVT2_1863','user_EVT2_1484','user_EVT3_2066','user_EVT3_3190','user_EVT2_1350','user_EVT3_3837','user_EVT2_1592',\
'user_EVT3_269','user_EVT3_4323','user_EVT3_2075','user_TCH_TYP_0','user_EVT2_1479','user_EVT3_3245','user_EVT3_3872','user_EVT3_4309','user_EVT3_3613','user_EVT3_4375','user_EVT2_115','user_EVT3_3810','user_EVT3_272','user_EVT2_1830','user_EVT3_3492',\
'user_EVT3_21','user_EVT3_1708','user_EVT2_1905','user_EVT3_4377','V22','user_EVT3_3617','user_EVT3_1711','user_EVT3_4346','user_EVT3_3834','user_EVT3_2073','user_EVT3_905','user_EVT3_3150','user_EVT2_1849','user_EVT3_3493','user_EVT3_3818','user_EVT3_4376',\
'user_EVT3_3796','user_EVT3_1201','user_EVT2_1264','user_EVT3_4277','user_EVT3_3756','user_EVT3_3188','user_EVT3_3809','user_EVT3_4312','user_EVT2_314','user_EVT3_3663','user_EVT3_1709','user_EVT3_3867','user_EVT3_3151','user_EVT3_3496','user_EVT3_4320',\
'user_EVT2_1268','user_EVT3_1691','user_EVT3_4270','user_EVT3_3784','user_EVT3_3174','user_EVT2_1858','user_EVT3_3664','user_EVT3_3759','user_EVT2_1857','user_EVT1_372','user_EVT3_19','user_EVT3_3772','user_EVT2_1909','user_EVT3_3797','user_EVT2_705','user_EVT2_222',\
'user_EVT3_3147','user_EVT2_1796','user_EVT2_1795','user_EVT2_701','user_EVT3_3745','user_EVT3_573','user_EVT2_557','user_EVT3_3802','user_day_26','user_EVT3_1698','user_EVT3_3130','user_EVT3_15','user_EVT3_889','user_EVT3_3248','V4','user_EVT3_2207','user_TCH_TYP_2',\
'user_EVT3_4273','user_EVT3_3161','user_EVT3_4332','user_EVT3_3778','user_EVT2_1588','user_EVT3_4292','user_EVT2_1859','user_EVT2_1906','user_EVT3_3672','user_EVT2_1860','user_EVT2_1263','user_EVT3_3870','user_EVT3_1717','user_EVT2_1265','user_day_3',\
'user_EVT3_4357','user_EVT3_2058','user_EVT3_3808','user_EVT2_392','user_EVT3_3817','user_EVT2_225','user_EVT2_703','user_day_8','user_EVT3_3859','user_EVT3_4286','user_EVT3_2044','user_EVT1_326','V13','user_EVT3_2047','user_EVT2_1853','user_EVT3_3791',\
'user_EVT3_3741','user_EVT3_4271','user_EVT1_518','user_EVT3_1678','user_EVT3_3752','user_EVT3_891','user_EVT3_1718','user_EVT3_1729','user_EVT3_567','user_EVT2_2165','user_EVT3_3926','user_EVT2_1845','user_EVT3_3924','user_EVT3_3854','user_EVT2_2142',\
'user_EVT3_3845','user_EVT2_228','user_day_23','user_day_20','user_EVT3_3742','user_EVT2_1838','user_EVT3_571','user_EVT2_393','V14','user_EVT3_2641','user_EVT3_3234','V16','V12','user_EVT2_706','user_EVT3_4327','user_EVT3_897','V8','user_EVT3_1699',\
'user_EVT2_1855','user_EVT3_3612','user_EVT3_1684','user_EVT3_275','user_EVT3_3640','user_EVT3_3724','user_EVT3_462','user_EVT2_2145','V20','user_EVT3_2064','V24','user_EVT3_4287','user_EVT2_230','user_EVT3_2078','user_EVT3_3779','user_EVT2_18','user_EVT3_3793',\
'user_EVT3_1728','user_EVT3_4278','user_EVT3_1683','user_EVT3_3758','user_EVT3_1697','user_EVT3_2032','user_EVT3_3743','user_EVT3_3135','user_EVT2_2161','user_EVT3_3811','user_EVT3_1707','user_EVT2_1590','user_EVT3_4296','user_EVT3_2057','user_EVT3_3781',\
'user_EVT3_3863','user_EVT3_3723','user_EVT3_3153','user_EVT2_1836','user_EVT3_4368','user_EVT3_4285','user_EVT2_1844','user_EVT3_3803','user_EVT3_4372','user_EVT3_4382','user_EVT3_4361','user_EVT3_2041','user_EVT3_4388','user_day_17','user_EVT3_3642',\
'user_EVT3_4308','user_EVT2_1262','gap_log_day_mean','V26','user_EVT3_3734','user_EVT3_1723','user_EVT3_4303','user_EVT3_3131','user_EVT3_3782','user_EVT3_3495','user_EVT3_4350','V18','user_EVT2_1866','user_EVT3_4386','user_EVT3_2054','user_EVT3_4370',\
'user_EVT3_4398','user_EVT3_1700','user_EVT3_2074','用户平均的访问时间','user_day_12','user_EVT1_438','user_EVT3_2053','user_EVT3_1685','user_EVT3_3853','user_EVT3_1686','user_EVT2_1349','user_EVT3_2051','user_EVT2_1799','user_EVT3_3795','user_EVT2_1041',\
'user_EVT3_3646','user_EVT3_4290','user_EVT3_3806','user_EVT3_1705','user_EVT3_4389','user_EVT2_2155','user_EVT3_4352','user_EVT2_1480','user_day_29','user_EVT3_3921','user_EVT3_4293','user_EVT3_4315','user_EVT3_4319','user_EVT2_1837','user_EVT3_1726',\
'user_EVT2_1589','user_EVT3_4395','user_EVT3_1701','user_EVT3_559','V19','user_EVT3_4325','user_EVT3_558','user_EVT3_1695','user_EVT3_1713','user_EVT1_257','user_EVT3_3831','user_EVT3_911','user_EVT2_922','user_EVT3_2033','user_EVT3_3622','user_EVT3_270',\
'user_EVT3_3821','user_EVT3_3850','user_EVT3_2005','user_EVT3_4307','user_EVT2_1912','user_EVT3_1725','user_EVT3_4363','user_EVT3_3157','user_EVT3_4362','user_EVT3_3497','user_EVT3_3671','user_EVT3_3812','user_EVT3_2063','user_EVT3_3616','user_EVT3_3792',\
'user_EVT2_1850','user_EVT3_2080','user_EVT3_3816','user_EVT2_1915','user_EVT3_3819','user_EVT3_3122','user_EVT3_3774','user_EVT3_3172','user_EVT3_2069','user_EVT3_3726','user_EVT2_2149','user_EVT3_3794','user_EVT3_1712','user_EVT3_3849','V21','user_EVT3_4331',\
'user_EVT3_3620','user_EVT3_3746','user_EVT2_1839','user_EVT3_2052','user_EVT3_4281','user_EVT2_394','user_EVT3_3777','user_EVT3_3925','user_EVT2_1908','V17','user_EVT3_2043','user_EVT3_3729','user_EVT1_540','user_EVT2_1864','user_EVT3_3231','user_EVT3_277',\
'user_EVT2_1865','user_EVT2_2159','user_EVT2_226','user_EVT3_3843','V2','V27','user_EVT3_3139','user_EVT2_1829','user_EVT3_3140','user_EVT3_3826','user_EVT3_3815','user_EVT3_4355','user_day_1','user_EVT2_17','user_EVT3_16','user_EVT3_2048','user_EVT3_4313',\
'user_EVT3_1196','user_EVT2_1046','user_EVT2_2133','user_EVT2_22','user_EVT3_3833','user_day_19','user_EVT3_276','user_EVT3_4276','user_EVT3_3790','user_day_13','user_EVT2_1827','user_EVT3_1680','user_EVT3_4384','user_EVT2_1861','user_EVT3_3700','user_EVT3_3768',\
'user_EVT3_4297','user_EVT2_223','user_EVT3_3839','user_EVT3_3143','V5','user_EVT3_1696','user_EVT3_4390','user_EVT3_562','user_EVT2_2167','user_EVT2_2160','user_EVT3_4333','user_EVT3_3783','user_EVT2_2134','user_EVT3_3858','user_EVT3_4394','user_EVT3_43',\
'user_EVT2_1266','V15','user_EVT3_3920','user_EVT2_2150','user_day_9','user_EVT3_3735','user_EVT3_910','user_EVT3_3171','user_EVT2_1862','user_EVT3_4305','user_EVT3_3498','user_EVT3_914','user_EVT3_3185','user_EVT3_3766','user_EVT3_3798','user_EVT2_1045',\
'user_EVT1_396','user_EVT1_508','user_EVT3_3780','user_EVT3_4300','user_EVT2_19','user_EVT2_1907','user_EVT3_4298','user_EVT2_569','user_EVT2_1267','用户最后的访问时间']
OFF_LINE = False

def xgb_model(train_set_x,train_set_y,test_set_x):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eta': 0.03,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 10,  # 2 3
              'silent':1,
              'eval_metric':'auc'
              }
    
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=490)
    importance = pd.Series(model.get_fscore()).sort_values(ascending=False)
    importance = pd.DataFrame(importance,columns = ['importance'])
    importance.reset_index(inplace=True)
    importance.columns = ['name','importance']
    print(type(importance))
    # plot_importance(model,max_num_features=20)
    # pyplot.show()
    tmp = columns_rank_list.copy()
    for i in range(1,31):
        tmp.remove(str('V'+str(i)))
    fsa = pd.DataFrame()
    fsa['name'] = tmp
    fsa['num'] = 1
    importance = pd.merge(importance, fsa, on=['name'])
    print(importance)
    importance = importance.set_index(['name'])
    importance = importance.iloc[:20,:]
    importance = importance.sort_values(by=['importance'],ascending=True)
    importance.loc[:,'importance'].plot.barh()
    plt.show()
    predict = model.predict(dvali)
    return predict

from math import log
def continue_day(dataframe):
    if len(set(list(dataframe))) <= 1:
        return -1
    day_list = sorted(list(set(list(dataframe))))
    con = 0
    maxx = 0
    for i in range(1,len(day_list)):
        if day_list[i-1] + 1 == day_list[i]:
            con += 1
        else:
            if maxx<con:
                maxx = con
            con = 0
    return maxx
        

def cal_ent(a_list):
    item_count = {}
    for item in a_list:
        if item not in item_count.keys():
            item_count[item] = 0
        item_count[item] += 1
    ent = 0.0
    for key in item_count:
        prob = float(item_count[key]) / len(a_list)
        ent -= prob * log(prob, 2)
    return ent
    

        
def log_tabel(original,dataframe):
    original_copy = original.copy()
    dataframe_copy = dataframe.copy()
    dataframe_copy['hour'] = dataframe_copy.OCC_TIM.map(lambda x:x.hour)
    dataframe_copy['day'] = dataframe_copy.OCC_TIM.map(lambda x:x.day)
    dataframe_copy['minute'] = dataframe_copy.OCC_TIM.map(lambda x:x.minute)
    dataframe_copy['EVT_LBL_1'] = dataframe_copy.EVT_LBL.map(lambda x:x.split('-')[0])
    dataframe_copy['EVT_LBL_2'] = dataframe_copy.EVT_LBL.map(lambda x:x.split('-')[1])
    dataframe_copy['EVT_LBL_3'] = dataframe_copy.EVT_LBL.map(lambda x:x.split('-')[2])

    tt = dataframe_copy.groupby(by=['USRID'])['day'].max().reset_index(name='用户最后的访问时间')
    original_copy = pd.merge(original_copy,tt,on=['USRID'],how='left')
#
    tt = dataframe_copy.groupby(by=['USRID'])['day'].mean().reset_index(name='用户平均的访问时间')
    original_copy = pd.merge(original_copy,tt,on=['USRID'],how='left')

    

    user_EVT3 = pd.crosstab(dataframe_copy.USRID,dataframe_copy.EVT_LBL_3)
    user_EVT3.columns = ['user_EVT3_'+str(i) for i in user_EVT3.columns]
    user_EVT3 = user_EVT3.reset_index()
    original_copy = pd.merge(original_copy,user_EVT3,on=['USRID'],how='left')
    user_EVT1 = pd.crosstab(dataframe_copy.USRID,dataframe_copy.EVT_LBL_1)
    user_EVT1.columns = ['user_EVT1_'+str(i) for i in user_EVT1.columns]
    user_EVT1 = user_EVT1.reset_index()
    original_copy = pd.merge(original_copy,user_EVT1,on=['USRID'],how='left')
    user_EVT2 = pd.crosstab(dataframe_copy.USRID,dataframe_copy.EVT_LBL_2)
    user_EVT2.columns = ['user_EVT2_'+str(i) for i in user_EVT2.columns]
    user_EVT2 = user_EVT2.reset_index()
    original_copy = pd.merge(original_copy,user_EVT2,on=['USRID'],how='left')


    user_TCH_TYP = pd.crosstab(dataframe_copy.USRID,dataframe_copy.TCH_TYP)
    user_TCH_TYP.columns = ['user_TCH_TYP_'+str(i) for i in user_TCH_TYP.columns]
    user_TCH_TYP = user_TCH_TYP.reset_index()
    original_copy = pd.merge(original_copy,user_TCH_TYP,on=['USRID'],how='left')
    
#    
#    
    user_day = pd.crosstab(dataframe_copy.USRID,dataframe_copy.day)
    user_day.columns = ['user_day_'+str(i) for i in user_day.columns]
    user_day = user_day.reset_index()
    original_copy = pd.merge(original_copy,user_day,on=['USRID'],how='left')
    
    gap_log_day_mean = pd.DataFrame(dataframe_copy.groupby(['USRID'])['day'].apply(lambda x: -20 if len(np.diff(np.array(sorted(set(x))),1))==0 else np.diff(np.array(sorted(set(x))),1).mean()))
    gap_log_day_mean = gap_log_day_mean.reset_index();gap_log_day_mean.columns = ['USRID','gap_log_day_mean']
    original_copy = pd.merge(original_copy,gap_log_day_mean,on=['USRID'],how='left')
    gap_log_day_min = pd.DataFrame(dataframe_copy.groupby(['USRID'])['day'].apply(lambda x: -20 if len(np.diff(np.array(sorted(set(x))),1))==0 else np.diff(np.array(sorted(set(x))),1).min()))
    gap_log_day_min = gap_log_day_min.reset_index();gap_log_day_min.columns = ['USRID','gap_log_day_min']
    original_copy = pd.merge(original_copy,gap_log_day_min,on=['USRID'],how='left')
    gap_log_day_max = pd.DataFrame(dataframe_copy.groupby(['USRID'])['day'].apply(lambda x: -20 if len(np.diff(np.array(sorted(set(x))),1))==0 else np.diff(np.array(sorted(set(x))),1).max()))
    gap_log_day_max = gap_log_day_max.reset_index();gap_log_day_max.columns = ['USRID','gap_log_day_max']
    original_copy = pd.merge(original_copy,gap_log_day_max,on=['USRID'],how='left')


    tt = dataframe_copy.groupby(by=['USRID','day'])['EVT_LBL'].count().reset_index(name='用户每天的访问量')
    tt['用户每天的访问量rank'] = tt.groupby(by=['USRID'])['用户每天的访问量'].rank(ascending=False)
    tt = tt[tt['用户每天的访问量rank']==1];tt = tt[['USRID','day']];tt.rename(columns = {'day':'用户在哪一天访问量最大'},inplace=True)
    original_copy = pd.merge(original_copy,tt,on=['USRID'],how='left')
    if dataframe_copy.shape[0]>300*10000:
        columns_rank_list.extend(['用户在哪一天访问量最大'])
    original_copy['用户在哪一天访问量最大减去最后访问时间'] = original_copy['用户在哪一天访问量最大'] - original_copy['用户最后的访问时间']
    original_copy['用户在哪一天访问量最大减去平均访问时间'] = original_copy['用户在哪一天访问量最大'] - original_copy['用户平均的访问时间']
    if dataframe_copy.shape[0]>300*10000:
        columns_rank_list.extend(['用户在哪一天访问量最大减去最后访问时间','用户在哪一天访问量最大减去平均访问时间'])
        
##############################################
    tt = dataframe_copy.groupby(by=['USRID'])['hour'].max().reset_index(name='用户最后的小时访问时间')
    original_copy = pd.merge(original_copy,tt,on=['USRID'],how='left')
    if dataframe_copy.shape[0]>300*10000:
        columns_rank_list.extend(['用户最后的小时访问时间'])
        
        
    


    return original_copy

if __name__ == '__main__':
    train_agg = pd.read_csv('../data/train/train_agg.csv',sep='\t')
    train_flg = pd.read_csv('../data/train/train_flg.csv',sep='\t')
    train_log = pd.read_csv('../data/train/train_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
    flg_agg_train = pd.merge(train_flg,train_agg,on=['USRID'],how='left')
    all_train = log_tabel(flg_agg_train,train_log)
    all_train.fillna(-999,inplace=True)
    if OFF_LINE == True:
        train_x = all_train.drop(['USRID', 'FLAG'], axis=1).values
        train_y = all_train['FLAG'].values
        auc_list = []
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=False)
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
        
    test_log = pd.read_csv('../data/test/test_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
    test_agg = pd.read_csv('../data/test/test_agg.csv',sep='\t')
    test_set = log_tabel(test_agg,test_log)
    test_set = test_set.fillna(-999)
    ##################################################################################
    result_name = test_set[['USRID']]
    train_x = all_train.drop(['USRID', 'FLAG'], axis=1)
    train_x = train_x.reindex_axis(columns_rank_list,axis=1)
    train_y = all_train['FLAG']
    test_x = test_set.drop(['USRID'], axis=1)
    
    test_x = test_x.reindex_axis(columns_rank_list,axis=1)

    # from sklearn import preprocessing
    # min_max_scaler = preprocessing.StandardScaler()
    # X_train_minmax = min_max_scaler.fit_transform(train_x)
    # X_test_minmax = min_max_scaler.transform(test_x)
    X_train_minmax = train_x
    X_test_minmax = test_x
    pred_result = xgb_model(X_train_minmax,train_y,X_test_minmax)
    result_name['RST'] = pred_result
    

    result_name.to_csv('chen_result.csv',index=None,sep='\t')