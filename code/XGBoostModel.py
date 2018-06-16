
# coding: utf-8

# In[2]:


#-*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import xgboost as xgb
import math
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV


# In[3]:


# calculate the time interval between start time and end time
def timeInterval(start_time, end_time):
    day1 = int(start_time[0:2])
    hour1 = int(start_time[2:4])
    minute1 = int(start_time[4:6])
    second1 = int(start_time[6:8])
    day2 = int(end_time[0:2])
    hour2 =  int(end_time[2:4])
    minute2 = int(end_time[4:6])
    second2 = int(end_time[6:8])
    if (day2 > day1):
        hour2 += (day2 - day1) * 24
    time_interval = (hour2  - hour1) * 3600 + (minute2 - minute1) * 60 + (second2 - second1)
    return time_interval


# In[4]:


# 起始时间是哪一天
def selectDay(start_time):
   day = int(start_time[0:2])
   return day


# In[5]:


# 起始时间是哪一小时
def selectHour(start_time):
    hour = int(start_time[2:4])
    return hour


# In[6]:


# voice data
voice_train = pd.read_csv('../data/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
voice_test = pd.read_csv('../data/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
voice = pd.concat([voice_train,voice_test],axis=0)
# 每次通话的时长（换算成秒）
voice['voice_time'] = voice.apply(lambda row: timeInterval(row['start_time'].zfill(8), row['end_time'].zfill(8)), axis=1)   
# 通话是哪一天开始
voice['voice_day'] = voice.apply(lambda row: selectDay(row['start_time'].zfill(8)), axis=1)   
#  通话是哪一小时开始
voice['voice_hour'] = voice.apply(lambda row: selectHour(row['start_time'].zfill(8)), axis=1)

voice.head()


# In[7]:


# message data
sms_train = pd.read_csv('../data/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
sms_test = pd.read_csv('../data/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
sms = pd.concat([sms_train,sms_test],axis=0)
# 短信是哪一天发送
sms['sms_day'] = sms.apply(lambda row: selectDay(row['start_time'].zfill(8)), axis=1)  
#  短信是哪一小时发送
sms['sms_hour'] = sms.apply(lambda row: selectHour(row['start_time'].zfill(8)), axis=1)  

sms.head()


# In[8]:


# website and app data
wa_train = pd.read_csv('../data/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})
wa_test = pd.read_csv('../data/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})
wa = pd.concat([wa_train,wa_test],axis=0)

wa.head()


# In[436]:


# train data
uid_train = pd.read_csv('../data/uid_train.txt',sep='\t',header=None,names=('uid','label'))

# test data
uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('../data/uid_test_b.txt',index=None)


# In[437]:


# voice feathers
voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index().fillna(0)

voice_opp_head = voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index().fillna(0)

voice_opp_len = voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)

voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

voice_time_invl_call_type = voice.groupby(['uid', 'call_type'])['voice_time'].sum().unstack().add_prefix('voice_time_invl_call_type_').reset_index().fillna(0)

voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)

voice_time_invl_in_out = voice.groupby(['uid', 'in_out'])['voice_time'].sum().unstack().add_prefix('voice_time_invl_in_out_').reset_index().fillna(0)


# In[518]:


voice_unique_day_call_type = voice.groupby(['uid', 'call_type'])['voice_day'].unique().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)
voice_unique_day_call_type.columns = ['uid', 'voice_unique_day_call_type_1', 'voice_unique_day_call_type_2',
                                     'voice_unique_day_call_type_3', 'voice_unique_day_call_type_4', 'voice_unique_day_call_type_5']
# print(voice_unique_day_call_type)

for i in range(45):
    for j in range(5):
        voice_unique_day_call_type["voice_day_"+str(i+1)+"_call_type_"+str(j+1)] = 0
print(voice_unique_day_call_type)

size = voice_unique_day_call_type.iloc[:,0].size
    
for i in range(size):
    for j in range(1, 6):
        temp_list = voice_unique_day_call_type.iloc[i, j]
        if (type(temp_list) == int and temp_list == 0):
            continue
#         elif (type(temp_list) == int and temp_list != 0):
#             temp_index = 5 * temp_list + j
#             voice_unique_day_call_type.iloc[i, temp_index] = 1
#             continue
        else:
            temp_len = len(temp_list)
            for k in range(temp_len):
                temp_index = 5 * temp_list[k] + j
                voice_unique_day_call_type.iloc[i, temp_index] = 1   # 有通话记录的天数，值设置为1
print(voice_unique_day_call_type)


# In[519]:


voice_unique_day_in_out = voice.groupby(['uid', 'in_out'])['voice_day'].unique().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)
voice_unique_day_in_out.columns = ['uid', 'voice_unique_day_in_out_0', 'voice_unique_day_in_out_1']
# print(voice_unique_day_in_out)

for i in range(45):
    for j in range(2):
        voice_unique_day_in_out["voice_day_"+str(i+1)+"_call_type_"+str(j)] = 0
print(voice_unique_day_in_out)

size = voice_unique_day_in_out.iloc[:,0].size
    
for i in range(size):
    for j in range(1, 3):
        temp_list = voice_unique_day_in_out.iloc[i, j]
        if (type(temp_list) == int and temp_list == 0):
            continue
#         elif (type(temp_list) == int and temp_list != 0):
#             temp_index = 2 * temp_list + j
#             voice_unique_day_in_out.iloc[i, temp_index] = 1
#             continue
        else:
            temp_len = len(temp_list)
            for k in range(temp_len):
                temp_index = 2 * temp_list[k] + j
                voice_unique_day_in_out.iloc[i, temp_index] = 1   # 有通话记录的天数，值设置为1
print(voice_unique_day_in_out)


# In[476]:


# new features of voice data

# # 平均每天的通话次数
# voice_day_call_freq = (voice.groupby(['uid'])['opp_num'].count()/45).reset_index().fillna(0)
# voice_day_call_freq.columns = ['uid','voice_day_call_freq']

# # 有通话记录的天数（总天数：45）
voice_day_num = voice.groupby(['uid'])['voice_day'].nunique().reset_index().fillna(0)

# 每个uid有通话记录是哪些天（list型数据）
voice_unique_day_num = voice.groupby(['uid'])['voice_day'].unique().reset_index().fillna(0)
# DataFrame型
voice_unique_day_num.columns = ['uid','voice_unique_day_num']

# 初始化为0（假设每个uid第01-45天都没有通话记录）
for i in range(45):
    voice_unique_day_num["voice_day_"+str(i+1)] = 0
    
size = voice_unique_day_num.iloc[:,0].size
 
# uid某一天有通话记录，则值设为1
for j in range(size):
    temp_list = voice_unique_day_num.iloc[j, 1]
    for k in range(len(temp_list)):
        temp_index = temp_list[k] + 1
        voice_unique_day_num.iloc[j, temp_index] = 1
        
print(voice_unique_day_num)


# In[ ]:


# 初始化为0（假设每个uid在第01-45天的通话次数都是0）
for i in range(45):
    voice_unique_day_num["voice_day_"+str(i+1)] = 0
 
size = voice_unique_day_num.iloc[:,0].size

# 每个uid在每一天的通话次数
temp = voice.groupby(['uid', 'voice_day'])['opp_num'].count().reset_index().fillna(0)
size1 = temp.iloc[:, 0].size

for i in range(size1):
    a = temp.iloc[i, 0]   # uid
    b = int(temp.iloc[i, 1]) + 1    #  有通话记录的天数
    c = temp.iloc[i, 2]   # 每天的通话次数
    for j in range(size):
        if (voice_unique_day_num.iloc[j, 0] == a):       # 将uid每天的通话次数写入对应的位置
            voice_unique_day_num.iloc[j, b] = c
            break
            
# print(voice_unique_day_num)  
# voice_unique_day_num.to_csv("../data/voice_unique_day_num.csv")


# In[ ]:


# # 数据归一化(通话的总次数、通话的不同对端个数)
# voice_opp_num_count_normalization = (voice_opp_num['voice_opp_num_count'] - voice_opp_num['voice_opp_num_count'].min())/(voice_opp_num['voice_opp_num_count'].max() - voice_opp_num['voice_opp_num_count'].min())
# voice_opp_num = voice_opp_num.drop("voice_opp_num_count", axis=1)
# voice_opp_num["voice_opp_num_count"] = voice_opp_num_count_normalization

# voice_opp_num_unique_count_normalization = (voice_opp_num['voice_opp_num_unique_count'] - voice_opp_num['voice_opp_num_unique_count'].min())/(voice_opp_num['voice_opp_num_unique_count'].max() - voice_opp_num['voice_opp_num_unique_count'].min())
# voice_opp_num = voice_opp_num.drop("voice_opp_num_unique_count", axis=1)
# voice_opp_num["voice_opp_num_unique_count"] = voice_opp_num_unique_count_normalization


# In[441]:


# 有通话记录的小时数
voice_hour_num = voice.groupby(['uid'])['voice_hour'].nunique().reset_index().fillna(0)

# 有通话记录是哪些小时
voice_unique_hour_num = voice.groupby(['uid'])['voice_hour'].unique().reset_index().fillna(0)
voice_unique_hour_num.columns = ['uid','voice_unique_hour_num']
# print(voice_unique_hour_num)

# 添加24列（不同的小时数）
size = voice_unique_hour_num.iloc[:,0].size
for i in range(23):
    voice_unique_hour_num["voice_hour_"+str(i+1)] = 0
    
for j in range(size):
    temp_list = voice_unique_hour_num.iloc[j, 1]
    for k in range(len(temp_list)):
        temp_index = temp_list[k] + 1
        voice_unique_hour_num.iloc[j, temp_index] = 1
# print(voice_unique_hour_num)


# In[442]:


# message features
sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index().fillna(0)

sms_opp_head = sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index().fillna(0)

sms_opp_len = sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)

sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)


# In[492]:


# new features of message data

# # 有短信记录的天数（总天数：45）
sms_day_num = sms.groupby(['uid'])['sms_hour'].nunique().reset_index().fillna(0)

# 有短信记录是哪些天
sms_unique_hour_num = sms.groupby(['uid'])['sms_hour'].unique().reset_index().fillna(0)
sms_unique_hour_num.columns = ['uid','sms_unique_hour_num']
# print(sms_unique_hour_num)

# 添加45列（不同的天数）
size = sms_unique_hour_num.iloc[:,0].size
for i in range(45):
    sms_unique_hour_num["sms_day_"+str(i+1)] = 0

for j in range(size):
    temp_list = sms_unique_hour_num.iloc[j, 1]
    for k in range(len(temp_list)):
        temp_index = temp_list[k] + 1
        sms_unique_hour_num.iloc[j, temp_index] = 1
print(sms_unique_hour_num)


# In[493]:


# 有短信记录的小时数
sms_hour_num = sms.groupby(['uid'])['sms_hour'].nunique().reset_index().fillna(0)

# 有短信记录是哪些小时
voice_unique_hour_num = voice.groupby(['uid'])['voice_hour'].unique().reset_index().fillna(0)
voice_unique_hour_num.columns = ['uid','voice_unique_hour_num']
# print(voice_unique_hour_num)

# 添加24列（不同的小时数）
size = voice_unique_hour_num.iloc[:,0].size
for i in range(23):
    voice_unique_hour_num["voice_hour_"+str(i+1)] = 0
    
for j in range(size):
    temp_list = voice_unique_hour_num.iloc[j, 1]
    for k in range(len(temp_list)):
        temp_index = temp_list[k] + 1
        voice_unique_hour_num.iloc[j, temp_index] = 1
# print(voice_unique_hour_num)


# In[444]:


# website and app features
wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index().fillna(0)

visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index().fillna(0)

visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index().fillna(0)

up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index().fillna(0)

down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index().fillna(0)


# In[ ]:


# new features of wa data

# all_flow = wa['up_flow'] + wa['down_flow']
# wa['all_flow'] = all_flow
# all_flow = wa.groupby(['uid'])['all_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_all_flow_').reset_index().fillna(0)


# In[445]:


# 每个uid有访问记录的总天数
wa_unique_day_num = wa.groupby(['uid'])['date'].unique().reset_index().fillna(0)
wa_unique_day_num.columns = ['uid','wa_unique_day_num']
# print(wa_unique_day_num)

# 给每个uid添加45列（代表总的45天）
for i in range(45):
    wa_unique_day_num["wa_day_"+str(i+1)] = 0
    
# # 添加45列（不同的天数）
size = wa_unique_day_num.iloc[:,0].size

for j in range(size):
    temp_list = wa_unique_day_num.iloc[j, 1]
    temp_length = len(temp_list)
    for k in range(temp_length):
        # 判断是不是nan
        if (math.isnan(float(temp_list[k]))):
            continue
        else:
            temp_index = int(temp_list[k]) + 1
            wa_unique_day_num.iloc[j, temp_index] = 1
# print(wa_unique_day_num)
# wa_unique_day_num.to_csv("../data/wa_unique_day_num.csv")


# In[536]:


# all features

# voice_unique_day_num = voice_unique_day_num.drop(['voice_unique_day_num'], axis = 1)

# voice_unique_hour_num = voice_unique_hour_num.drop(['voice_unique_hour_num'], axis = 1)

# voice_unique_day_call_type = voice_unique_day_call_type.drop(['voice_unique_day_call_type_1', 'voice_unique_day_call_type_2',
#          'voice_unique_day_call_type_3', 'voice_unique_day_call_type_4', 'voice_unique_day_call_type_5'], axis = 1)

# voice_unique_day_in_out = voice_unique_day_in_out.drop(['voice_unique_day_in_out_0', 'voice_unique_day_in_out_1'], axis = 1)

# sms_unique_day_num = sms_unique_day_num.drop(['sms_unique_day_num'], axis = 1)

# sms_unique_hour_num = sms_unique_hour_num.drop(['sms_unique_hour_num'], axis = 1)

# wa_unique_day_num = wa_unique_day_num.drop(['wa_unique_day_num'], axis = 1)

# voice_day_call_number = pd.read_csv('../data/voice_unique_day_num.csv')

# wa_day_visit_number = pd.read_csv('../data/wa_unique_day_num.csv')

feature = [voice_opp_num, voice_unique_day_num, voice_call_type, voice_in_out, 
           voice_time_invl_call_type, voice_time_invl_in_out, voice_opp_head, voice_opp_len, 
           sms_opp_num, sms_unique_day_num, sms_unique_hour_num, sms_opp_head, sms_opp_len, sms_in_out,
           wa_name, visit_cnt, visit_dura, up_flow, down_flow]


# In[537]:


# train data (uid, label and features)
train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')
    
# test data (uid and features)
test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')
    
# write the result of features
train_feature.to_csv('../data/result/train_feature_new_v2.csv', index=None)
test_feature.to_csv('../data/result/test_feature_new_v2.csv', index=None)


# In[538]:


# 自定义评价函数
def evalMetric(preds,dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds, 'label':label})
    pre= pre.sort_values(by='preds',ascending=False)

    auc = metrics.roc_auc_score(pre.label, pre.preds)
    pre.preds = pre.preds.map(lambda x: 1 if x>=0.5 else 0)
    f1 = metrics.f1_score(pre.label, pre.preds)

    res = 0.6*auc + 0.4*f1
    return 'res', res

dtrain = xgb.DMatrix(train_feature.drop(['uid', 'label'], axis = 1), label = train_feature.label)
dtest = xgb.DMatrix(test_feature.drop(['uid'], axis = 1))

# the parameters of lightgbm model
xgb_params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'learning_rate': 0.08,
    'max_depth': 5,
    'min_child_weight': 10
}

# local cross validation
xgb.cv(xgb_params,dtrain,num_boost_round=200,nfold=3,verbose_eval=5,early_stopping_rounds=180,maximize=True,feval=evalMetric)

# train the model
model = xgb.train(xgb_params,dtrain=dtrain,num_boost_round=200,verbose_eval=5,evals=[(dtrain,'train')],maximize=True,feval=evalMetric,early_stopping_rounds=250)

# predict the test data
pred = model.predict(dtest)
res = pd.DataFrame({'uid':test_feature.uid, 'label':pred})
res = res.sort_values(by='label',ascending=False)
res.label = res.label.map(lambda x: 1 if x>=0.5 else 0)
res.label = res.label.map(lambda x: int(x))

res.to_csv('../data/result/xgb_result_new_v4.csv', index=False, header=False, sep=',', columns=['uid','label'])

