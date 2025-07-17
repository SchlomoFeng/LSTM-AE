from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame as pd_DataFrame
from dill import dump, load
import redis
import os
import re
import requests
import json
from numpy import nan, isnan, array, inf, nanmax,nanmin
import requests
from math import isinf
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

import random


def read_redis(host: str, port: str, db: str, group_id: str, tag_list: list, password: str = '', start: int = 0, end: int = -1):
    '''
    从redis读取指定位号数据
    :param host:
    :param port:
    :param db:
    :param password:
    :param group_id: 位号组id
    :param tag_list: 位号列表
    :param start:
    :param end:
    :return:
    '''
    pool = redis.ConnectionPool(host=host, port=port, db=db, password=password, decode_responses=True)
    client = redis.Redis(connection_pool=pool)
    ls = [client.lrange(f'data:tag_value:group:{group_id}:{tag_id}', start, end) for tag_id in tag_list]
    df = pd.concat([pd.DataFrame([eval(eval(i)) for i in tag]) for tag in ls])
    if len(df) == 0:
        assert False, (', '.join(tag_list) + '不存在')
    del df['tag_time']
    df.rename(columns={'app_time': 'tag_time'}, inplace=True)
    df = df[['tag_time', 'tag_name', 'tag_value']]
    # df['tag_value'] = df['tag_value'].astype('float')
    df['tag_value'] = pd.to_numeric(df['tag_value'], errors='coerce')
    df.drop_duplicates(subset=['tag_time', 'tag_name'], keep='first', inplace=True)  # 去除重复行
    df = pd.pivot(df, values='tag_value', index='tag_time', columns='tag_name')
    df.dropna(axis=0, how='any', inplace=True)  # 去掉空值
    df.index = pd.to_datetime(df.index)
    df = df.sort_values('tag_time')
    client.close()
    return df


def dataFilterByQuery(data_df: pd.DataFrame, filter_condition: str):
     
    numerical_cols = data_df.columns
    virtual_cols = [f'tag_{index + 1}' for index in range(len(numerical_cols))]
    col_map, col_map_inverse = {}, {}
    for key, value in zip(numerical_cols, virtual_cols):
        col_map[key] = value
        col_map_inverse[value] = key

    pattern = r"VAL\(\"|\"\)"  # 匹配位号
    filter_condition = re.sub(pattern, "", filter_condition)
    for index, col in enumerate(numerical_cols):
        filter_condition = filter_condition.replace(col, virtual_cols[index])

    data_df_ = data_df.rename(columns=col_map)
    if filter_condition == '':  # 如果前置条件为空
        data_df_ = data_df.copy()
    else:
        data_df_ = data_df_.query(filter_condition).copy()
    data_df_.rename(columns=col_map_inverse, inplace=True)
    return data_df_
    



def IQR(data, high_quantile=0.99, low_quantile=0.01, scale=4):
    '''

    Parameters
    ----------
    data: 位号数据
    high_quantile: 高置信度
    low_quantile：低置信度
    scale： 放缩系数

    Returns
    -------

    '''    
    high_quantile_value, avg_value, low_quantile_value = data.quantile([high_quantile, 0.5, low_quantile]).values
    quantile = max((high_quantile_value - avg_value), (avg_value - low_quantile_value))
    dynamic_upper_limit = avg_value + quantile * scale
    dynamic_lower_limit = avg_value - quantile * scale
    return quantile, dynamic_upper_limit, dynamic_lower_limit

def getHisDataByApi(url, tagNames, beginTime, endTime, interval=30):
    '''

    Parameters
    ----------
    url
    tagNames: 位号列表
    startTime: 开始时间
    endTime：结束时间
    interval： 采样间隔

    Returns
    -------

    '''
    headers = {
        "Connection": "keep-alive",
        "Content-Type": "application/json",
    }
    request_info = {
        "data": {
            "begTime": beginTime,
            "endTime": endTime,
            "interval": interval,
            "isSecond": True,
            "option": 0,
            "tagNames": tagNames
        }
    }
    # assert False, tagNames
    r_json = requests.post(url, json=request_info, headers=headers, verify=False, timeout=20)
    result = json.loads(r_json.content.decode())
    data_df = pd.DataFrame(result['content']['records'])[['tagName', 'tagValue', 'tagTime']]
    return data_df

def alarm_judge(last_ts: pd.DataFrame, warning_tag: str, craft_lower_limit: float, craft_upper_limit: float):
    """

    Parameters
    ----------

    warning_value : float
        预警值
    craft_lower_limit : float or int
        工艺下限
    craft_upper_limit : float or int
        工艺上限
    Returns
    -------
    warning_status: 预警状态
    warning_reasons: 预警原因
    """

    # 0. 初始化
    warning_time = last_ts.index.max()
    warning_value = last_ts.loc[warning_time, warning_tag]

    # 1. 参数处理及合法性判断
    # 1.1如果工艺上下限为空，为了不影响后续判断，将其认为为正无穷或者负无穷。
    craft_lower_limit = float('-inf') if pd.isna(craft_lower_limit) else craft_lower_limit  # 当工艺上限为空，用正无穷代替
    craft_upper_limit = float('inf') if pd.isna(craft_upper_limit) else craft_upper_limit  # 当工艺下限为空，用负无穷代替

    # 1.2参数合法性判断：工艺上限必须大于工艺下限
    if craft_lower_limit > craft_upper_limit:
        warning_reasons = f'工艺下限{craft_lower_limit:.2f}必须小于或等于工艺上限{craft_upper_limit:.2f}'
        warning_status = 10
    else:
        # 2. 报警上下限判断
        if warning_value > craft_upper_limit:
            warning_status = 30
            warning_reasons = f'实时值高于工艺上限{craft_upper_limit};'
        elif warning_value < craft_lower_limit:
            warning_status = 30
            warning_reasons = f'实时值低于工艺下限{craft_lower_limit};'
        else:
            warning_status = 10
            warning_reasons = ''
    return warning_status, warning_reasons







def main(warningItemId: str, warnTags: list, upper_limit: float, lower_limit: float, inputTagData: list):
    '''

    :param warning_tag: 报警位号
    :param warning_tag: 前置条件
    :param filter_config: 滤波配置
    :param alarm_config: 报警上下限配置
    :return:
    '''
    # 0. 初始化
    ## 开始计时
    time_start = datetime.now()  


    # 数据处理
    warningTags = list(inputTagData.keys())

    data_list = []
    for key in inputTagData:
        # key = "NB.LJSJ.LIC_2A340A.PV"
        tmp = pd.DataFrame(inputTagData[key])
        tmp["tagName"] = key
        data_list.append(tmp)
    data_df = pd.concat(data_list, axis=0)
    data_df = data_df.drop_duplicates(['appTime','tagName'])
    data_df["appTime"] = pd.to_datetime(data_df["appTime"])
    # data_df.set_index(keys="appTime", inplace=True)
    # data_df.head()
    data_df['tagValue'] = pd.to_numeric(data_df['tagValue'], errors='coerce')

    data_df = pd.pivot(data_df, values='tagValue', index='appTime', columns='tagName')



    tagDetail = []
    warningTime = data_df.index.max()
    for warningTag in  warningTags:
        warning_value = data_df.loc[warningTime, warningTag]

        if lower_limit < warning_value < upper_limit:
            warningStatus = 10
        else:
            warningStatus = 20
        tagDetail.append({"warningTime": warningTime.strftime('%Y-%m-%d %H:%M:%S'), 
                          "warningTag": warningTag, 
                          "tagValue": warning_value,
                          "expectValue": warning_value + 0.001,
                          "predictValue": [],
                          "warningStatus": warningStatus,
                          "upperLimit": upper_limit,
                           "lower_limit": lower_limit})
     

    warning_result_df = {
        "warningItemId": warningItemId, # 预警项id
        "warningTime": time_start.strftime('%Y-%m-%d %H:%M:%S'), # 预警时间
        "warningStatus": random.choice([10, 20, 30, 40]), # 预警项状态，10代表正常，20代表通知，30代表预警，40代表报警
        "warningReason": "快速上升预警", # 预警项原因，
        "warningTags": ",".join(warningTags), # 预警异常的位号
        "otherInfo": "tag1,tag2", # 备用字段,其他业务spd所需信息
        "feedbackInfo": "{\"tag\": 1}", # 回写位号信息
        "tagDetail": tagDetail  
    }
    warning_result = json.dumps(warning_result_df)
    print({'warning_result': warning_result})
    return {'warning_result': warning_result}


if __name__ == '__main__':

    inputTagData = {"NB.LJSJ.LIC_2A340A.PV":[
                {
                "tagValue": 58.89362,
                "tagTime": "2024-08-16 09:14:59",
                "appTime": "2024-08-16 09:15:00",
                "quality": 192
                },
                {
                "tagValue": 58.907925,
                "tagTime": "2024-08-16 09:15:29",
                "appTime": "2024-08-16 09:15:30",
                "quality": 192,
                }
            ]
        }
    
    warningItemId = 1234566
    warnTags = ['tag1', 'tag2']
    lower_limit = 1
    upper_limit = 2


    lower_limit = "1"
    upper_limit = "2"

    main(warningItemId, warnTags, upper_limit, lower_limit, inputTagData)

    # inputParameter =  {"warningItemId":10,"lower_limit":10,"upper_limit":100,"warnTags":["TAG1","TAG2","NB.LJSJ.LIC_2A340A.PV","HGLZMDCGL01.EL02_147_UMEAS_CALC"],"inputTagData":{"HGLZMDCGL01.EL02_147_UMEAS_CALC":[{"appTime":"2024-08-28T12:48:54","tagValue":"2.94522476","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:49:24","tagValue":"2.94628549","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:49:54","tagValue":"2.94575644","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:50:28","tagValue":"2.94551969","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:50:58","tagValue":"2.945948","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:51:38","tagValue":"2.945961","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:52:08","tagValue":"2.945578","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:52:48","tagValue":"2.94616961","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:53:18","tagValue":"2.94530964","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:53:49","tagValue":"2.945817","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:54:19","tagValue":"2.946207","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:54:49","tagValue":"2.945256","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:55:29","tagValue":"2.94596982","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:55:59","tagValue":"2.94565153","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:56:39","tagValue":"2.94566536","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:57:19","tagValue":"2.94564056","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:57:59","tagValue":"2.9459","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:58:29","tagValue":"2.94561362","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:59:09","tagValue":"2.94508028","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T12:59:49","tagValue":"2.94576454","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:00:33","tagValue":"2.94585752","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:01:03","tagValue":"2.945913","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:01:33","tagValue":"2.94523382","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:02:03","tagValue":"2.94562364","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:02:49","tagValue":"2.94544435","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:03:19","tagValue":"2.94565","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:03:49","tagValue":"2.945737","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:04:29","tagValue":"2.94556022","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:05:09","tagValue":"2.945741","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:05:39","tagValue":"2.94558573","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:06:19","tagValue":"2.9454987","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:06:49","tagValue":"2.94571877","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:07:22","tagValue":"2.9450922","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:08:02","tagValue":"2.94562244","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:08:42","tagValue":"2.94540215","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:09:22","tagValue":"2.94570684","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:09:52","tagValue":"2.94553757","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:10:32","tagValue":"2.94553447","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:11:18","tagValue":"2.946074","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:11:57","tagValue":"2.94564","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:12:44","tagValue":"2.945794","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:13:14","tagValue":"2.94550633","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:13:44","tagValue":"2.94595432","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:14:22","tagValue":"2.945372","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:14:52","tagValue":"2.94575119","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:15:24","tagValue":"2.945583","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:16:08","tagValue":"2.944688","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:16:44","tagValue":"2.94549537","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:17:17","tagValue":"2.94567633","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:17:57","tagValue":"2.94496655","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:18:27","tagValue":"2.94531655","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192},{"appTime":"2024-08-28T13:19:08","tagValue":"2.94575334","tagTime":null,"tagName":"HGLZMDCGL01.EL02_147_UMEAS_CALC","quality":192}]}}
    # main(**inputParameter)





