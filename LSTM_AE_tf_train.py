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


class LSTMAETraining:
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

    def create_autoencoder_model(self, X_shape):
        """创建LSTM自编码器模型"""
        inputs = Input(shape=(X_shape[1], X_shape[2]))
        L1 = LSTM(32, activation='relu', return_sequences=True,
                  kernel_regularizer=regularizers.l2(0.01))(inputs)
        L2 = LSTM(16, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X_shape[1])(L2)
        L4 = LSTM(16, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(32, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(X_shape[2]))(L5)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mae')
        return model

    def prepare_sequences(self, data, sequence_length=10):
        """将连续时序数据转换为滑窗样本"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])   # 滑窗大小 = sequence_length
        return np.array(sequences)

    def save_model_components(self):
        """保存模型、标准化器和阈值"""
        model_dir = '/data'
        os.makedirs(model_dir, exist_ok=True)

        # 保存Keras模型
        model_path = os.path.join(model_dir, f'lstm_ae_{self.warningItemId}.h5')
        self.model.save(model_path)
        # Keras的model.save()只能保存到本地文件系统路径，不能直接保存到S3文件系统对象
        # 解决方法：本地文件系统 → 临时文件 → S3存储

        with tempfile.NamedTemporaryFile(suffix = '.h5', delete = False) as tmp_file:
            try:
                self.model.save(tmp_file.name)

                with open(tmp_file.name, 'rb') as local_file:
                    with fs.open(model_path, 'wb') as s3_file:
                        s3_file.write(local_file.read())
                    
                    tmp_file.flush()    # 强制系统将临时文件写入硬盘中
                    self.model = load_model(tmp_file.name)
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)    # 删除临时文件
            

        # 保存标准化器和阈值
        components_path = os.path.join(model_dir, f'components_{self.warningItemId}.pkl')
        components = {
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_columns': self.feature_columns
        }
        with fs.open(components_path, 'wb') as s:
            dill.dump(components, s)    # 与pickle类似, 用于序列化模型

        print(f"模型组件已保存到: {model_path} 和 {components_path}")

    def train_model(self, train_data):
        """训练模型"""
        # 数据标准化
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(train_data)

        # 滑窗大小
        sequence_length = 10
        X_train = self.prepare_sequences(scaled_data, sequence_length)

        # 创建和训练模型
        self.model = self.create_autoencoder_model(X_train.shape)
        history = self.model.fit(X_train, X_train,
                                 epochs=20,
                                 batch_size=32,
                                 validation_split=0.1,
                                 verbose=0)

        # 计算异常阈值
        X_pred_train = self.model.predict(X_train, verbose=0)
        train_mae = np.mean(np.abs(X_pred_train - X_train), axis=(1, 2))
        self.threshold = np.mean(train_mae) + 3 * np.std(train_mae)

        # 保存所有模型组件
        self.save_model_components()

        return history, train_mae


def main(warningItemId: str = 'lstm_ae_demo', is_train: bool = True):
    """
    主函数：根据参数执行训练、预警或追溯功能
    """
    time_start = datetime.now()

    # 初始化预警系统
    warning_system = LSTMAETraining(warningItemId)

    
    # 执行训练算法
    print("开始训练LSTM-AE模型...")

    # 加载数据
    train_data, _ = warning_system.load_data()
    history, train_mae = warning_system.train_model(train_data)

    # 生成训练结果用于可视化
    train_result = pd.DataFrame({
        'timestamp': train_data.index[:len(train_mae)],
        'reconstruction_error': train_mae,
        'threshold': warning_system.threshold,
        'is_anomaly': train_mae > warning_system.threshold
    })

    # 生成训练报告
    train_report = pd.DataFrame([{
        'model_type': 'LSTM-AutoEncoder',
        'train_samples': len(train_data),
        'threshold': warning_system.threshold,
        'anomaly_rate': np.mean(train_mae > warning_system.threshold),
        'final_loss': history.history['loss'][-1]
    }])

    return {
        'modelUpdateTime': time_start.strftime('%Y-%m-%d %H:%M:%S'),
        'trainResult': train_result,
        'trainReport': train_report
    }
