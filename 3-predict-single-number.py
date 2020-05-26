import cv2
import numpy as np
from keras.datasets import mnist

import keras
import matplotlib.pyplot as plt
import matplotlib
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
    # new_array = cv2.resize(img_array, (img_size, img_size))
    
    
    
    new_array = new_array.reshape(1, img_size, img_size, 1)
    new_array = new_array.astype("float32")
    new_array /= 255
    prtary(np.squeeze(new_array))
    # shows(ret)
    return new_array


model = keras.models.load_model('recognition2.model')

prediction = model.predict([prepare('tester4.jpg')])

print(prediction)
print(np.argmax(prediction))
exit()


(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
img_row,img_col,channel = 28,28,1

mnist_input_shape = (img_row,img_col,1)

train_images = train_images.reshape(train_images.shape[0],img_row,img_col,channel)
test_images = test_images.reshape(test_images.shape[0],img_row,img_col,channel)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

# myshow(train_images[0])
# myshowlist(train_images, train_labels)

## 进行归一化处理
train_images  /= 255
test_images /= 255

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
            plt.imshow(np.squeeze(images[index]),cmap='gray')
    plt.show()

show_images(train_images, train_labels)
# prtary(train_images[50000])

# 将类向量，转化为类矩阵
# 从 5 转换为 0 0 0 0 1 0 0 0 0 0 矩阵
test_labels = keras.utils.to_categorical(test_labels,10)

prediction = model.predict(test_images)

for i in range(10):
    print(np.argmax(prediction[i]))