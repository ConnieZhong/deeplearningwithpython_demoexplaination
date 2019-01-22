import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

from keras.datasets import imdb
from keras import models
from keras import layers
from keras import backend as K
from keras import preprocessing
from keras.layers import Flatten, Dense, Embedding
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.optimizers import RMSprop


#原始数据 
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay -1
    i = min_index + lookback
    while 1:

        rows = np.random.randint(min_index + lookback, max_index, size = batch_size)
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size = batch_size)
        else:
            if i+batch_size >= max_index:
                i=min_index+lookback
                rows = np.arrange(i, min(i+batch_size, max_index))
                i+= len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
#进行配置，使用50%的GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )


#从文本中读取数据
data_dir='/data/ceph/conniezhong/deeplearning/dataset/'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)

#解析数据
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines): #遍历并且有下标生成
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values


lookback = 1440
step = 6
delay = 128
batch_size = 128
train_gen = generator(float_data, lookback=lookback,delay=delay, min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step, batch_size=batch_size)
test_gen = generator(float_data,lookback=lookback,delay=delay, min_index=300001, max_index=None, step=step, batch_size=batch_size)
val_step = (300000-200001 - lookback) // batch_size
test_step = (len(float_data) - 300001 - lookback) // batch_size


#构建顺序网络
model = models.Sequential()
model.add(layers.GRU(32, input_shape = (None, float_data.shape[-1]))) #10000是标记的个数，8是嵌入的维度, input_len 是输入的单词的长度
model.add(Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')

        
model.summary()

history = model.fit_generator(train_gen, steps_per_epoch=500, epochs = 20, validation_data=val_gen, validation_steps=val_step)

#

print(history)

