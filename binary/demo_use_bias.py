import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

from keras.datasets import imdb
from keras import models
from keras import layers
from keras import backend as K

def vec_se(seqs, demi = 10000):
    results = np.zeros((len(seqs), demi))
    for i,seq in enumerate(seqs):
         results[i,seq] = 1.
    return results
         

# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
#进行配置，使用50%的GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )

(train_data, train_labels) ,(test_data, test_labels) = imdb.load_data(num_words=10000)

#将输入数据变成矩阵
x_train = vec_se(train_data)
x_test = vec_se(test_data)

#将标签变成向量
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#构建顺序网络
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape=(10000,), use_bias=True))
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

#编译模型
model.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#将最后的10000个作为验证集
x_val = x_train[:5000] 
y_val = y_train[:5000]

#将最后的10000个标签作为验证集中的标签
partial_x_train = x_train[5000:]
partial_y_train = y_train[5000:]
#partial_x_train_v = K.variable(partial_x_train)

print('train',len(x_train)) #25000

#partial_y_train_v = K.variable(partial_y_train)

#history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#history = model.fit(partial_x_train_v, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val))



print(train_data[0])
