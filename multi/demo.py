import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

from keras import models
from keras import layers
from keras import backend as K
from keras.datasets import reuters

def vec_se(seqs, demi = 10000):
    results = np.zeros((len(seqs), demi))
    for i,seq in enumerate(seqs):
         results[i,seq] = 1.
    return results
         

# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
config = tf.ConfigProto()
#进行配置，使用50%的GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)

(org_train_data, org_train_labels) ,(org_test_data, org_test_labels) = reuters.load_data(num_words=10000)

#将输入数据变成矩阵
train_data = vec_se(org_train_data)
test_data = vec_se(org_test_data)

#将标签变成矩阵
train_labels = vec_se(org_train_labels,demi = 46)
test_labels = vec_se(org_test_labels,demi = 46)

print('===================')

#构建顺序网络
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

#编译模型
model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#将最后的10000个作为验证集
val_data = train_data[:300] 
val_labels= train_labels[:300]

#将最后的10000个标签作为验证集中的标签
real_train_data = train_data[300:]
real_train_labels = train_labels[300:]
#partial_x_train_v = K.variable(partial_x_train)

#print('train',len(x_train)) #25000

#partial_y_train_v = K.variable(partial_y_train)

#history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
history = model.fit(real_train_data, real_train_labels, epochs=20, batch_size=512, validation_data=(val_data, val_labels))
#history = model.fit(partial_x_train_v, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val))

results = model.evaluate(test_data, test_labels)
print('result:', results)


print(train_data[0])
