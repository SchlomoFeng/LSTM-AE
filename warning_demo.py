# -*- coding:utf-8 -*-

import redis
import pandas as pd
import numpy as np
from datetime import datetime
# import connectorx as cx
# from sklearn.metrics import r2_score
# from dill import load
import os
import json
import random
import json

def read_redis(group_id: str, tag_list: list):
    """
    数据读取
    Parameters
    ----------

    group_id : 组id
    tag_list : 位号列表(list(str))
    Returns
    -------
    df: 数据
    """
    # pool = redis.ConnectionPool(host='hzsh-ibd.supcon5t.com', port='30379', db='1', password='', decode_responses=True)
    pool = redis.ConnectionPool(host='gpt-demo2.supcon5t.com', port='30379', db='1', password='', decode_responses=True)

    client = redis.Redis(connection_pool=pool)
    ls = [client.lrange(f'data:tag_value:group:{group_id}:{tag_id}', -5, -1) for tag_id in tag_list]
    df = pd.concat([pd.DataFrame([eval(eval(i)) for i in tag]) for tag in ls])
    if len(df) == 0:
        assert False, (', '.join(tag_list) + '不存在')
    del df['tag_time']
    df.rename(columns={'app_time': 'tag_time'}, inplace=True)
    df = df[['tag_time', 'tag_name', 'tag_value']]
    df['tag_value'] = df['tag_value'].astype('float')
    #     df = pd.pivot_table(df, values='tag_value', index='tag_time', columns='tag_name', aggfunc=np.mean).reset_index()
    # df = pd.pivot_table(df, values='tag_value', index='tag_time', columns='tag_name', aggfunc=np.mean)
    df.drop_duplicates(subset=['tag_time', 'tag_name'], keep='first', inplace=True)  # 去除重复行
    df = pd.pivot(df, values='tag_value', index='tag_time', columns='tag_name')
    df.index = pd.to_datetime(df.index)
    df = df.sort_values('tag_time')
    client.close()
    return df


def main(warningItemId: str, is_backtest: bool, test_time: dict, warning_tag: str, related_tag: list, is_train: bool=True, is_evaluate: bool=True):
    '''
    测试算法
    :param modelId: 模型id
    :param train_time: 测试时间
    :param warning_tag: 预警位号
    :param related_tag: 关联位号
    :param is_train: 是否训练
    :param is_evaluate: 是否评估
    :return:
    '''


    if warning_tag == '':
        warning_tag = 'A234.PV'

    # 加载训练算法得到的模型

    # try:
    #     model_path = os.path.join('./root/app/model/lr', warningItemId) + '.pkl'
    #     with open(model_path, "rb") as f:
    #         model = load(f)
    # except:
    #     model_path = os.path.join('/root/app/model/lr', warningItemId) + '.pkl'
    #     with open(model_path, "rb") as f:
    #         model = load(f)

    if is_backtest == False:  # 执行预警算法
        time_start = datetime.now()
        warning_result = {
            "warningItemId": warningItemId, # 预警项id
            "warningTime": time_start.strftime('%Y-%m-%d %H:%M:%S'), # 预警时间
            "warningStatus": random.choice([10, 20, 30, 40]), # 预警项状态，10代表正常，20代表通知，30代表预警，40代表报警
            "warningReason": "快速上升预警", # 预警项原因，
            "warningTags": warning_tag, # 预警异常的位号
            "otherInfo": "tag1,tag2", # 备用字段,其他业务spd所需信息
            "tagDetail": [{
                        "warningTime": time_start.strftime('%Y-%m-%d %H:%M:%S'), 
                        "warningTag": warning_tag, 
                        "warningStatus": random.choice([10, 20, 30, 40]),  # 预警项状态，10代表正常，20代表通知，30代表预警，40代表报警
                        "warningReason": "快速上升预警", # 预警项原因，
                        "tagValue": 1,
                        "upperLimit": 1.7, 
                        "lowerLimit": 1.0, 
                        "predictValue": [{
                            "tagTime": time_start.strftime('%Y-%m-%d %H:%M:%S'), 
                            "tagValue": 1, 
                            "upperLimit": 1.7, 
                            "lowerLimit": 0.5},
                            {
                            "tagTime": time_start.strftime('%Y-%m-%d %H:%M:%S'), 
                            "tagValue": 1, 
                            "upperLimit": 1.7, 
                            "lowerLimit": 0.5
                            }]              
                        }]   
        }

        # predLen = 60
        # future_df = pd.DataFrame({"tag_time": pd.date_range(start=datetime.now(), periods=predLen, freq='30s').strftime('%Y-%m-%d %H:%M:%S'), 
        #                         'predict_value': [50] * predLen,  
        #                         'upper_limit': [50] * predLen, 
        #                         'lower_limit': [50] * predLen,
        #                         # 'tag_value': [None] * predLen,
        #                         'tag_status': [random.choice([10, 20, 30]) for _ in range(predLen)],
        #                         })
        
        # future_result = future_df.to_dict(orient='records')

        # warning_result_df['tagDetail'][0]['predictValue'] = future_result



        warningResult = json.dumps(warning_result)
        # assert False, warning_result_df
        return {'warningResult': warningResult}


    else:  # 否则，执行追溯算法
        # test_time = {'timeType': 'fixedtime', 'timeInterval': '36000'}
        # warning_tag = 'TC20702.PV'
        # related_tag = ['TC20702.SV']

        # key = {'driver': 'pymysql', "host": "gateway.supcon5t.com", "username": "root", "password": "Supcon1304",
        #        "database": "ibd-warning", "port": '13306'}
        # conn = 'mysql://' + key['username'] + ':' + key['password'] + '@' + key['host'] + ':' + key['port'] + '/' + key[
        #     'database']  # connection token
        # if test_time['timeType'] == 'fixedtime':
        #     # max_time_trace = int(int(test_time['timeInterval'])/60)
        #     max_time_trace = int(int(test_time['timeInterval']))

        # table = cx.read_sql(conn,
        #                     f'select warn_time, tag_id, tag_value, warn_status from tb_log_202307 where task_id = {modelId} and warn_time between date_sub(Now(), interval ' +
        #                     str(max_time_trace) + ' Hour) and Now()',
        #                     return_type="arrow")  # or arrow2 https://github.com/jorgecarleitao/arrow2
        # warning_history_df = table.to_pandas(split_blocks=False, date_as_object=False)
        # if warning_history_df.loc[:, ['warn_time', 'tag_id']].duplicated().any():
        #     warning_history_df.drop_duplicates(subset=['warn_time', 'tag_id'], keep='first', inplace=True)  # 去除重复行
        # warning_history_df = pd.pivot(warning_history_df, index='warn_time', columns='tag_id',
        #                               values='tag_value')  # 长表转宽表
        # warning_history_df.dropna(axis=0, how='any', inplace=True)  # 去掉空值

        # X_test = warning_history_df[related_tag].values
        # y_test = warning_history_df[warning_tag].values
        # y_test_predict = model.predict(X_test)
        # tags = related_tag + [warning_tag]
        # raw_df = pd.DataFrame(np.c_[X_test, y_test], columns=tags, index=warning_history_df.index)
        # raw_df.reset_index(inplace=True)
        # raw_df = pd.melt(raw_df, id_vars=['warn_time'], var_name='tag_id', value_name='tag_value')
        # predict_df = pd.DataFrame(np.c_[np.full(X_test.shape, np.nan), y_test], columns=tags, index=warning_history_df.index)
        # predict_df.reset_index(inplace=True)
        # predict_df = pd.melt(predict_df, id_vars=['warn_time'], var_name='tag_id', value_name='predict_value')
        # test_df = pd.merge(left=raw_df, right=predict_df, on=['warn_time', 'tag_id'])

        # # 计算测试评价结果
        # r2 = r2_score(y_test, y_test_predict)
        report_df = pd.DataFrame([{'r2': 0.05}])
        # print('拟合优度为', r2)




        time_start = datetime.now()
        test_list = []

        for _ in range(10):
            warning_result = {
                "warningItemId": warningItemId, # 预警项id
                "warningTime": time_start.strftime('%Y-%m-%d %H:%M:%S'), # 预警时间
                "warningStatus": random.choice([10, 20, 30, 40]), # 预警项状态，10代表正常，20代表通知，30代表预警，40代表报警
                "warningReason": "快速上升预警", # 预警项原因，
                "warningTags": "tag1,tag2", # 异常位号
                "otherInfo": "任意信息 ", # 备用字段,其他业务spd所需信息
                "tagDetail": [{
                            "warningTime": time_start.strftime('%Y-%m-%d %H:%M:%S'), 
                            "warningTag": warning_tag, 
                            "warningStatus": random.choice([10, 20, 30, 40]),  # 预警项状态，10代表正常，20代表通知，30代表预警，40代表报警
                            "warningReason": "快速上升预警", # 预警项原因，
                            "tagValue": 1,
                            "upperLimit": 1.7, 
                            "lowerLimit": 1.0, 
                            "expectedValue": 1.0, 
                            "predictValue": [{
                                "tagTime": time_start.strftime('%Y-%m-%d %H:%M:%S'), 
                                "tagValue": 1, 
                                "upperLimit": 1.7, 
                                "lowerLimit": 0.5},
                                {
                                "tagTime": time_start.strftime('%Y-%m-%d %H:%M:%S'), 
                                "tagValue": 1, 
                                "upperLimit": 1.7, 
                                "lowerLimit": 0.5
                                }]              
                            
                            }]   
            }
            time_start += pd.Timedelta(1, unit='min')
            # warning_result = json.dumps(warning_result_df)
            test_list.append(pd.DataFrame(warning_result))
        test_df = pd.concat(test_list, axis=0)

        print(test_df, report_df)
        return {'warningResult': test_df, 'warningReport': report_df}


if __name__ == '__main__':
    warningItemId = '1673931259061080064'
    test_time = {'timeType': 'fixedtime', 'timeInterval': '36000'}
    warning_tag = 'TC20702.PV'
    related_tag = ['TC20702.SV']
    # is_backtest = True
    is_backtest = True
    main(warningItemId, is_backtest, test_time, warning_tag, related_tag)


