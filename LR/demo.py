import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

from keras import models
from keras import layers
from keras import backend as K
from keras.datasets import boston_housing

def build_model():
    #构建顺序网络
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu', input_shape=(org_train_data.shape[1],)))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))

    #编译模型
    model.compile(optimizer = 'rmsprop', loss='mse', metrics=['accuracy'])
    return model

# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
config = tf.ConfigProto()
#进行配置，使用50%的GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)

(org_train_data, org_train_labels) ,(org_test_data, org_test_labels) = boston_housing.load_data()

mean = org_train_data.mean(axis=0)
org_train_data -= mean
std = org_train_data.std(axis=0)
org_train_data /= std

org_test_data -= mean
org_test_data /= std

print('===================')

k = 4 
num_val_samples = len(org_train_data) // k 
num_epochs = 100
all_score = []
for i in range(k):
    print('processing turn:', i)
    val_data = org_train_data[i*num_val_samples : (i+1)*num_val_samples]
    val_target = org_train_labels [i*num_val_samples : (i+1)*num_val_samples]

    partial_train_data = np.concatenate([
        org_train_data[:i*num_val_samples],
        org_train_data[(i+1)*num_val_samples:]],axis=0)


    partial_train_target = np.concatenate([
        org_train_labels[:i*num_val_samples],
        org_train_labels[(i+1)*num_val_samples:]],axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_target, epochs=num_epochs,
            batch_size=1,verbose=1)
    val_mse, val_mae = model.evaluate(val_data, val_target, verbose=1)
    all_score.append(val_mae)

print('result:', all_score)
