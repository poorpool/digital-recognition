import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import matplotlib

matplotlib.use('TkAgg')

# 将数据集中图片展示出来
def show_images(images, labels):
    n = 8
    m = 8
    fig = plt.figure()
    for i in range(n):
        for j in range(m):
            index = i*n+j
            plt.subplot(n,m,index+1)
            plt.subplots_adjust(wspace=0.2, hspace=0.8)
            plt.title(labels[index])
            plt.imshow(images[index],cmap='gray')
    plt.show()

# 加载数据
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

#将数据维度进行处理
img_row,img_col = 28,28
train_images = train_images.reshape(train_images.shape[0],img_row,img_col).astype("float32")

print(train_images[0])

show_images(train_images, train_labels)