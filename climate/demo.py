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


# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
#进行配置，使用50%的GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )

max_feature = 10000
maxlen = 500

(x_train, y_train) ,(x_test, y_test) = imdb.load_data(num_words=max_feature)

#不足的长度补位，超过的进行截断
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen = maxlen)


#构建顺序网络
model = models.Sequential()
#每个文章有20个单词，每个单词是一个8维的嵌入，一个文章就表示为了一个20*8的向量
model.add(Embedding(10000, 32)) #10000是标记的个数，8是嵌入的维度, input_len 是输入的单词的长度
#将每个文章的20*8的向量平面化成为一个一维的向量
model.add(SimpleRNN(32))
#最后再经过一个sigmod激活，over
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

results = model.evaluate(x_test, y_test)

print(results)

