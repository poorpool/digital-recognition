import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('TkAgg')

# 定义常量
batch_size = 32
num_classes = 10
img_row, img_col, channel = 28, 28, 1

# 引入MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
mnist_input_shape = (img_row, img_col, 1)

# 将数据维度进行处理
train_images = train_images.reshape(train_images.shape[0], img_row, img_col, channel).astype("float32")
test_images = test_images.reshape(test_images.shape[0], img_row, img_col, channel).astype("float32")

## 进行归一化处理
train_images  /= 255
test_images /= 255

# 独热编码
# 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

# 构造网络结构

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=mnist_input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=mnist_input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))  # Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合
model.add(Flatten())  # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# 模型示意图
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=300)

# 模型编译
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# 进行训练
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=10, verbose=1,
                    validation_split=0.25, validation_data=(test_images,test_labels), shuffle=True)

score = model.evaluate(test_images,test_labels,verbose=1)

# 绘制图表
plt.plot(history.history['accuracy'])
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 保存训练完毕的模型
model.save('recognition.model')