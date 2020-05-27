import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Convolution2D
import cv2
import matplotlib
matplotlib.use('TkAgg')

batch_size=32
num_classes=10

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

print(train_images.shape,train_labels.shape)
print(test_images.shape,test_labels.shape)

"""
将数据集中图片展示出来
"""

def show_mnist(train_image,train_labels):
    n = 3
    m = 3
    fig = plt.figure()
    for i in range(n):
        for j in range(m):
            plt.subplot(n,m,i*n+j+1)
            plt.subplots_adjust(wspace=0.2, hspace=0.8)
            index = i * n + j #当前图片的标号
            img_array = train_image[index]
            img = Image.fromarray(img_array)
            plt.title(train_labels[index])
            plt.imshow(img,cmap='Greys')
    plt.show()

def myshow(image):
    im = plt.imshow(np.squeeze(image),cmap='gray')
    plt.show()

def myshowlist(images, labels):
    n = 8
    m = 8
    fig = plt.figure()
    for i in range(n):
        for j in range(m):
            plt.subplot(n,m,i*n+j+1)
            plt.subplots_adjust(wspace=0.2, hspace=0.8)
            # index = i * n + j #当前图片的标号
            # img_array = train_image[index]
            # img = Image.fromarray(img_array)
            plt.title(labels[i*n+j+1])
            plt.imshow(np.squeeze(images[i*n+j+1]),cmap='gray')
    plt.show()

img_row,img_col,channel = 28,28,1

mnist_input_shape = (img_row,img_col,1)

#将数据维度进行处理
train_images = train_images.reshape(train_images.shape[0],img_row,img_col,channel)
test_images = test_images.reshape(test_images.shape[0],img_row,img_col,channel)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

# myshow(train_images[0])
# myshowlist(train_images, train_labels)

## 进行归一化处理
train_images  /= 255
test_images /= 255

# 将类向量，转化为类矩阵
# 从 5 转换为 0 0 0 0 1 0 0 0 0 0 矩阵
train_labels = keras.utils.to_categorical(train_labels,num_classes)
test_labels = keras.utils.to_categorical(test_labels,num_classes)





"""
构造网络结构
"""

""" model 1
model = Sequential()


model.add(Conv2D(32,kernel_size=(3,3),
                    activation="relu",
                    input_shape=mnist_input_shape))
                    # kernalsize = 3*3 并没有改变数据维度
model.add(Conv2D(16,kernel_size=(3,3),
                    activation="relu"
                    ))
model.add(MaxPooling2D(pool_size=(2,2)))
                    # 进行数据降维操作
model.add(Flatten())#Flatten层用来将输入“压平”，即把多维的输入一维化，
                    #常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Dense(32,activation="relu"))
                    #全连接层
model.add(Dense(num_classes,activation='softmax'))

编译网络模型,添加一些超参数


model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
"""
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=mnist_input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))  # Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合

model.add(Flatten())  # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# 模型编译
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])






model.fit(train_images,
            train_labels,
            batch_size=batch_size,
            epochs=5,
            verbose=1,
            validation_data=(test_images,test_labels),
            shuffle=True
            )

score = model.evaluate(test_images,test_labels,verbose=1)

print('test loss:',score[0])
print('test accuracy:',score[1])

model.save('recognition.model')