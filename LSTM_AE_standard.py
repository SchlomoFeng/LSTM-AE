# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import json
import pickle
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


class LSTMAEWarningSystem:
    def __init__(self, warningItemId):
        self.warningItemId = warningItemId
        self.model = None
        self.scaler = None
        self.threshold = None
        self.feature_columns = ['YT.11FI_02044.PV', 'YT.11PIC_02044.PV', 'YT.11TI_02044.PV']

    def load_data(self):
        """加载数据并进行预处理"""
        try:
            data = pd.read_csv('Dataset/气化一期S4_imputed.csv')
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
        """准备时序数据"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

    def save_model_components(self):
        """保存模型、标准化器和阈值"""
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)

        # 保存Keras模型
        model_path = os.path.join(model_dir, f'lstm_ae_{self.warningItemId}.h5')
        self.model.save(model_path)

        # 保存标准化器和阈值
        components_path = os.path.join(model_dir, f'components_{self.warningItemId}.pkl')
        components = {
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_columns': self.feature_columns
        }
        with open(components_path, 'wb') as f:
            pickle.dump(components, f)

        print(f"模型组件已保存到: {model_path} 和 {components_path}")

    def load_model_components(self):
        """加载模型、标准化器和阈值"""
        model_dir = 'model'
        model_path = os.path.join(model_dir, f'lstm_ae_{self.warningItemId}.h5')
        components_path = os.path.join(model_dir, f'components_{self.warningItemId}.pkl')

        if os.path.exists(model_path) and os.path.exists(components_path):
            # 加载Keras模型
            self.model = load_model(model_path)

            # 加载标准化器和阈值
            with open(components_path, 'rb') as f:
                components = pickle.load(f)

            self.scaler = components['scaler']
            self.threshold = components['threshold']
            self.feature_columns = components['feature_columns']

            print("模型组件加载成功")
            return True
        else:
            print("未找到已保存的模型组件")
            return False

    def train_model(self, train_data):
        """训练模型"""
        # 数据标准化
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(train_data)

        # 创建时序数据
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

    def detect_anomaly(self, data):
        """异常检测"""
        if self.model is None or self.scaler is None or self.threshold is None:
            raise ValueError("模型或标准化器未初始化")

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
         warning_tag: str = '', related_tag: list = None,
         is_train: bool = False, is_evaluate: bool = True):
    time_start = datetime.now()

    # 初始化预警系统
    warning_system = LSTMAEWarningSystem(warningItemId)

    if is_train:
        # 执行训练算法
        print("开始训练LSTM-AE模型...")

        # 加载数据
        train_data, test_data = warning_system.load_data()
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

    elif is_backtest:
        # 执行追溯算法
        print("开始执行追溯测试...")

        # 尝试加载已训练模型
        if not warning_system.load_model_components():
            print("未找到已训练模型，先执行训练...")
            train_data, test_data = warning_system.load_data()
            warning_system.train_model(train_data)
        else:
            # 如果模型加载成功，只需要加载测试数据
            _, test_data = warning_system.load_data()

        # 使用测试数据进行异常检测
        is_anomaly, anomaly_score, mae_loss = warning_system.detect_anomaly(test_data)
        #print("test_samples:", len(test_data))
        #print("\nanomaly_count:", np.sum(is_anomaly))
        #print("\nanomaly_rate:", np.mean(is_anomaly))
        #print("\navg_reconstruction_error:", np.mean(mae_loss))
        #print("\nthreshold:", warning_system.threshold)
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

        # 尝试加载已训练模型
        if not warning_system.load_model_components():
            print("未找到已训练模型，先执行训练...")
            train_data, test_data = warning_system.load_data()
            warning_system.train_model(train_data)
        else:
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


if __name__ == '__main__':
    # 演示训练功能
    print("=== 演示训练功能 ===")
    train_result = main(
        warningItemId='lstm_ae_demo_001',
        is_train=True,
        is_backtest=False
    )
    print("训练完成！")
    print("训练报告:", train_result['trainReport'].to_dict('records')[0])

    print("\n=== 演示追溯功能 ===")
    backtest_result = main(
        warningItemId='lstm_ae_demo_001',
        is_train=False,
        is_backtest=True,
        test_time={'timeType': 'fixedtime', 'timeInterval': '24'}
    )
    print("追溯测试完成！")
    print("测试报告:", backtest_result['warningReport'].to_dict('records')[0])
    print(f"检测到 {len(backtest_result['warningResult'])} 个时间点的预警结果")

    print("\n=== 演示实时预警功能 ===")
    realtime_result = main(
        warningItemId='lstm_ae_demo_001',
        is_train=False,
        is_backtest=False
    )
    print("实时预警完成！")
    warning_data = json.loads(realtime_result['warningResult'])
    print(f"预警状态: {warning_data['warningStatus']}")
    print(f"预警原因: {warning_data['warningReason']}")