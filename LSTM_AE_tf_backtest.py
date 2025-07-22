# -*- coding: utf-8 -*-
import os
# import sys
# sys.path.append("/root/app/model/")
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dill
import s3fs
fs = s3fs.S3FileSystem()
import tempfile

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.models import Model, load_model
from keras import regularizers

import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)


class LSTMAEBacktest:
    def __init__(self, warningItemId):
        self.warningItemId = warningItemId
        self.model = None
        self.scaler = None
        self.threshold = None
        self.feature_columns = ['YT.11FI_02044.PV', 'YT.11PIC_02044.PV', 'YT.11TI_02044.PV']

    def load_data(self):
        """加载数据并进行预处理"""
        try:
            data = pd.read_csv(fs.open('/data/气化一期S4_imputed.csv', 'rb'))
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.index = data['timestamp']
            total_data = data.loc[:, "YT.11FI_02044.PV":"YT.11TI_02044.PV"]

            # 分割训练和测试数据
            total_len = len(total_data)
            train_ratio = 0.8
            split_idx = int(total_len * train_ratio)
            train_data = total_data[:split_idx]
            test_data = total_data[split_idx:]

            return train_data, test_data
        except Exception as e:
            # 如果文件不存在，生成模拟数据
            print(f"数据文件不存在，生成模拟数据: {e}")
            return self.generate_mock_data()

    def generate_mock_data(self):
        """生成模拟数据用于演示"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        n_samples = len(dates)

        # 生成正常数据
        normal_flow = 50 + 10 * np.sin(np.linspace(0, 10 * np.pi, n_samples)) + np.random.normal(0, 2, n_samples)
        normal_pressure = 100 + 5 * np.cos(np.linspace(0, 8 * np.pi, n_samples)) + np.random.normal(0, 1, n_samples)
        normal_temp = 80 + 15 * np.sin(np.linspace(0, 12 * np.pi, n_samples)) + np.random.normal(0, 3, n_samples)

        # 在最后20%数据中引入异常
        anomaly_start = int(0.8 * n_samples)
        normal_flow[anomaly_start:] += np.random.normal(10, 5, n_samples - anomaly_start)
        normal_pressure[anomaly_start:] += np.random.normal(-15, 3, n_samples - anomaly_start)

        data = pd.DataFrame({
            'YT.11FI_02044.PV': normal_flow,
            'YT.11PIC_02044.PV': normal_pressure,
            'YT.11TI_02044.PV': normal_temp
        }, index=dates)

        # 分割数据
        split_idx = int(0.8 * len(data))
        return data[:split_idx], data[split_idx:]

    def load_model_components(self):
        """加载模型、标准化器和阈值"""
        model_dir = '/data'
        model_path = os.path.join(model_dir, f'lstm_ae_194.h5')     # 硬编码
        components_path = os.path.join(model_dir, f'components_194.pkl')    # 硬编码

        try:
            if fs.exists(model_path) and fs.exists(components_path):
                # 加载Keras模型
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                    try:
                    # 加载标准化器和阈值
                        with fs.open(model_path, 'rb') as s3_file:
                            tmp_file.write(s3_file.read())

                        tmp_file.flush()
                        self.model = load_model(tmp_file.name)
                    finally:
                        if os.path.exists(tmp_file.name):
                            os.unlink(tmp_file)
                
                with fs.open(components_path, 'rb') as s:
                    components = dill.load(s)   # 与pickle类似, 用于反序列化模型

                self.scaler = components['scaler']
                self.threshold = components['threshold']
                self.feature_columns = components['feature_columns']

                print("模型组件加载成功")
                return True
            else:
                print("未找到已保存的模型组件")
                return False
        except Exception as e:
            print(f"从S3加载模型组件失败: {e}")
            return False

    def detect_anomaly(self, data):
        """异常检测"""
        if self.scaler is None or self.threshold is None:
            raise ValueError("标准化器和阈值未初始化")
        if self.model is None:
            raise ValueError("模型未初始化")

        # 数据预处理
        scaled_data = self.scaler.transform(data)
        sequence_length = 10

        if len(scaled_data) < sequence_length:
            # 如果数据不足，用最后一行重复填充
            padding = np.tile(scaled_data[-1], (sequence_length - len(scaled_data), 1))
            scaled_data = np.vstack([scaled_data, padding])

        X_test = self.prepare_sequences(scaled_data, sequence_length)

        # 预测和计算重构误差
        X_pred = self.model.predict(X_test, verbose=0)
        mae_loss = np.mean(np.abs(X_pred - X_test), axis=(1, 2))

        # 判断异常
        is_anomaly = mae_loss > self.threshold
        anomaly_score = mae_loss / self.threshold

        return is_anomaly, anomaly_score, mae_loss


def main(warningItemId: str, is_backtest: bool = True, test_time: dict = None,
         warning_tag: str = '', related_tag: list = None):
    time_start = datetime.now()

    # 初始化
    warning_system = LSTMAEBacktest(warningItemId)

    if is_backtest == True:
        # 执行追溯算法
        print("开始执行追溯测试...")
        
        # 如果模型加载成功，只需要加载测试数据
        _, test_data = warning_system.load_data()

        # 使用测试数据进行异常检测
        is_anomaly, anomaly_score, mae_loss = warning_system.detect_anomaly(test_data)

        # 生成追溯结果
        backtest_results = []
        for i, (timestamp, row) in enumerate(test_data.iterrows()):
            if i < len(is_anomaly):
                warning_status = 30 if is_anomaly[i] else 10  # 30=预警, 10=正常
                warning_reason = f"重构误差异常: {mae_loss[i]:.4f}" if is_anomaly[i] else "正常"

                warning_result = {
                    "warningItemId": warningItemId,
                    "warningTime": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    "warningStatus": warning_status,
                    "warningReason": warning_reason,
                    "warningTags": ",".join(warning_system.feature_columns),
                    "otherInfo": f"异常得分: {anomaly_score[i]:.4f}",
                    "tagDetail": [{
                        "warningTime": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        "warningTag": warning_system.feature_columns[0],
                        "tagValue": float(row[warning_system.feature_columns[0]]),
                        "upperLimit": float(warning_system.threshold),
                        "lowerLimit": 0.0,
                        "predictValue": []
                    }]
                }
                backtest_results.append(warning_result)

        # 生成测试报告
        warning_report = pd.DataFrame([{
            'test_samples': len(test_data),
            'anomaly_count': np.sum(is_anomaly),
            'anomaly_rate': np.mean(is_anomaly),
            'avg_reconstruction_error': np.mean(mae_loss),
            'threshold': warning_system.threshold
        }])

        # 转换为DataFrame格式用于展示
        backtest_df = pd.DataFrame(backtest_results)

        return {
            'warningResult': backtest_df,
            'warningReport': warning_report
        }

    else:
        # 执行实时预警算法
        print("执行实时预警...")
        
        # 如果模型加载成功，只需要加载测试数据用于模拟实时数据
        _, test_data = warning_system.load_data()

        # 使用最新数据进行异常检测
        recent_data = test_data.tail(20)  # 取最近20个数据点
        is_anomaly, anomaly_score, mae_loss = warning_system.detect_anomaly(recent_data)

        # 获取最新状态
        latest_anomaly = is_anomaly[-1] if len(is_anomaly) > 0 else False
        latest_score = anomaly_score[-1] if len(anomaly_score) > 0 else 0
        latest_timestamp = recent_data.index[-1]
        latest_values = recent_data.iloc[-1]

        warning_status = 30 if latest_anomaly else 10
        warning_reason = f"LSTM-AE异常检测: 重构误差{mae_loss[-1]:.4f}" if latest_anomaly else "正常运行"

        warning_result = {
            "warningItemId": warningItemId,
            "warningTime": time_start.strftime('%Y-%m-%d %H:%M:%S'),
            "warningStatus": warning_status,
            "warningReason": warning_reason,
            "warningTags": ",".join(warning_system.feature_columns),
            "otherInfo": f"异常得分: {latest_score:.4f}",
            "tagDetail": []
        }

        # 添加每个传感器的详细信息
        for tag in warning_system.feature_columns:
            tag_detail = {
                "warningTime": latest_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "warningTag": tag,
                "tagValue": float(latest_values[tag]),
                "upperLimit": float(warning_system.threshold),
                "lowerLimit": 0.0,
                "predictValue": []
            }
            warning_result["tagDetail"].append(tag_detail)

        return {'warningResult': json.dumps(warning_result)}
