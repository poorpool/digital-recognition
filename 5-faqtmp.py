import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
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

print("shape0 ", train_images.shape[0])

def prtary(mat):
    for i in range(len(mat)):
        for k in range(len(mat[i])):
            if (mat[i][k]<0.01):
                print("    ", end=" ")
            else:
                print('%.2f'%mat[i][k], end=" ")
        print()
prtary(train_images[0])
