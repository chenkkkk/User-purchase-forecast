# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:26:21 2018

@author: CCL
"""

import pandas as pd

ans1 = pd.read_csv("chen_result.csv",sep='\t',engine='python')
ans1.columns = ['USRID','RST_c']


ans2 = pd.read_csv("gao.csv",sep='\t',engine='python')
ans2.columns = ['USRID','RST_g']

ans3 = pd.read_csv("zhao.csv",sep='\t',engine='python')
ans3.columns = ['USRID','RST_z']

result = pd.merge(ans1,ans2,on='USRID')
result = pd.merge(result,ans3,on='USRID')


result['RST'] = 0.5*(0.35*result['RST_z'] + 0.65*result['RST_g'])+0.5*result['RST_c']
#
result = result[['USRID','RST']].copy()
################################
def is_nan(x):
    if str(x) == 'nan':
        return 1
    else:
        return 0
ans = result.copy()
test_log = pd.read_csv('../data/test/test_log.csv',sep='\t')


test_log = test_log.drop_duplicates(subset=['USRID'])

result = pd.merge(ans,test_log,on=['USRID'],how='left')

result['is_nan'] = result.EVT_LBL.map(lambda x:1 if str(x)=='nan' else 0)

result.loc[result.is_nan==1,'RST'] = result.loc[result.is_nan==1,'RST']*1.35
result = result[['USRID','RST']].copy()

#########################

test_log = pd.read_csv('../data/test/test_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
test_log['hour'] = test_log.OCC_TIM.map(lambda x:int(x.hour))
test_log['day'] = test_log.OCC_TIM.map(lambda x:int(x.day))


def func(x):
    if 18 in list(x):
        return 1
    else:
        return 0
ee = test_log.groupby(by=['USRID'])['hour'].apply(func).reset_index(name = '用户访问的小时')


test_flag = result.copy()
#
test_result = pd.merge(test_flag,ee,on=['USRID'],how='left')

test_result.loc[test_result['用户访问的小时']==1,'RST'] = test_result.loc[test_result['用户访问的小时']==1,'RST']*1.1

test_result[['USRID','RST']].to_csv('big_fusion_7_14_ratio1.35_18_1.1.csv',index=None,sep='\t')