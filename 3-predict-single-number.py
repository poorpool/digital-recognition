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


model = keras.models.load_model('recognition2.model')

prediction = model.predict([prepare('tester4.jpg')])

print(prediction)
print(np.argmax(prediction))


