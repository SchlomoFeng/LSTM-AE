#%%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.externals import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
# matplotlib inline

from numpy.random import seed
#from tensorflow import set_random_seed
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

# set random seed
seed(10)
tf.random.set_seed(10)
#%%
data = pd.read_csv('data/气化一期S4_imputed.csv')
#%%
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.index = data['timestamp']
# data = data.sort_index()
#%%
print("Dataset shape:", data.shape)
data.head()
#%%
total_data = data.loc[:,"YT.11FI_02044.PV":"YT.11TI_02044.PV"]
total_data
#%%
# 分割数据 - 确保有足够的数据进行训练
total_len = len(total_data)
train_ratio = 0.8
split_idx = int(total_len * train_ratio)
train = total_data[:split_idx]
test = total_data[split_idx:]
print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)
#%%
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot((data['YT.11FI_02044.PV'] - data['YT.11FI_02044.PV'].mean()) / data['YT.11FI_02044.PV'].std(), label='flow', color='blue', animated = True, linewidth=1)
ax.plot((data['YT.11PIC_02044.PV'] - data['YT.11PIC_02044.PV'].mean()) / data['YT.11PIC_02044.PV'].std(), label='pressure', color='red', animated = True, linewidth=1)
ax.plot((data['YT.11TI_02044.PV'] - data['YT.11TI_02044.PV'].mean()) / data['YT.11TI_02044.PV'].std(), label='temperature', color='green', animated = True, linewidth=1)
# ax.plot(train['Bearing 4'], label='Bearing 4', color='black', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Training Data', fontsize=16)
plt.show()
#%%
# normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
#scaler_filename = "scaler_data"
#joblib.dump(scaler, scaler_filename)
#%%
# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)
#%%
# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(32, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(32, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model
#%%
# create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()
#%%
# fit the model to the data
nb_epochs = 15
batch_size = 128
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.1).history
#%%
# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()
#%% md
# # Distribution of Loss Function
# By plotting the distribution of the calculated loss in the training set, one can use this to identify a suitable threshold value for identifying an anomaly. In doing this, one can make sure that this threshold is set above the “noise level” and that any flagged anomalies should be statistically significant above the background noise.
#%%
# plot the loss distribution of the training set
X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns,index=train.index)

scored_train = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored_train['Loss_mae'], bins = 20, kde= True, color = 'blue');
plt.xlim([-0.1,.3])
#%% md
# From the above loss distribution, let's try a threshold value of 0.275 for flagging an anomaly. We can then calculate the loss in the test set to check when the output crosses the anomaly threshold.
#%%
k = 3  
threshold = np.mean(scored_train['Loss_mae']) + k * np.std(scored_train['Loss_mae'])
print(f"Calculated Threshold: {threshold}")
#%%
scored_train['Threshold'] = threshold
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
#%%
X_pred_test = model.predict(X_test)
X_pred_test = X_pred_test.reshape(X_pred_test.shape[0], X_pred_test.shape[2])
X_pred_test = pd.DataFrame(X_pred_test, columns=test.columns, index=test.index)

scored_test = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored_test['Loss_mae'] = np.mean(np.abs(X_pred_test - Xtest), axis=1)
#%% md
# Having calculated the loss distribution and the anomaly threshold, we can visualize the model output in the time leading up to the bearing failure.
#%%
scored_test['Threshold'] = threshold
scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']

scored_final = pd.concat([scored_train, scored_test])
scored_final.plot(logy=True, figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red','green'])
#%% md
# This analysis approach is able to flag the upcoming bearing malfunction well in advance of the actual physical failure. It is important to define a suitable threshold value for flagging anomalies while avoiding too many false positives during normal operating conditions.
#%%
# save all model information, including weights, in h5 format
model.save("Cloud_model.h5")
print("Model saved")
#%%
