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
import random
import math
matplotlib.use('TkAgg')

def shows(images):
    plt.title(1)
    plt.imshow(np.squeeze(images),cmap='gray')
    plt.show()
def prtary(mat):
    for i in range(len(mat)):
        for k in range(len(mat[i])):
            if (mat[i][k]<0.01):
                print("    ", end=" ")
            else:
                print('%.2f'%mat[i][k], end=" ")
        print()
def prepare(path):
    img_size = 28
    new_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ret, binary = cv2.threshold(new_array,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)# 二值化
    # print(ret)
    # print(binary)
    new_array = binary
    new_array = new_array.reshape(1, img_size, img_size, 1)
    new_array = new_array.astype("float32")
    new_array /= 255

    

    # prtary(np.squeeze(new_array))
    # shows(ret)
    return new_array


model = keras.models.load_model('recognition.model')

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

img_row,img_col,channel = 28,28,1

mnist_input_shape = (img_row,img_col,1)
# ret, train_images = cv2.threshold(train_images,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)# 二值化
# ret, test_images = cv2.threshold(test_images,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)# 二值化
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

train_labels = keras.utils.to_categorical(train_labels,10)
test_labels = keras.utils.to_categorical(test_labels,10)

def drawDigit3(position, image, title, isTrue):
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap='gray_r')
    plt.axis('off')
    if not isTrue:
        plt.title(title, color='red')
    else:
        plt.title(title)
        
def batchDraw3(batch_size, test_X, test_y):
    selected_index = random.sample(range(len(test_y)), k=batch_size)
    images = test_X[selected_index]
    labels = test_y[selected_index]
    predict_labels = model.predict(images)
    image_number = images.shape[0]
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number
    plt.figure(figsize=(row_number+8, column_number+8))
    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index+1)
                image = images[index]
                actual = np.argmax(labels[index])
                predict = np.argmax(predict_labels[index])
                isTrue = actual==predict
                title = 'actual:%d, predict:%d' %(actual,predict)
                drawDigit3(position, image, title, isTrue)

batchDraw3(49, test_images, test_labels)
plt.show()

