"""x_train: uint8 NumPy array of grayscale image data with shapes (60000, 28, 28), containing the training data. Pixel values range from 0 to 255.

y_train: uint8 NumPy array of digit labels (integers in range 0-9) with shape (60000,) for the training data.

x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), containing the test data. Pixel values range from 0 to 255.

y_test: uint8 NumPy array of digit labels (integers in range 0-9) with shape (10000,) for the test data."""
from array import array
from itertools import count
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = tf.keras.utils.normalize(train_X, axis = 1)
test_X = tf.keras.utils.normalize(test_X, axis = 1)

def calculate_dist(x,y): 
    
    #dist = np.sqrt(np.sum(np.power(2,np.subtract(y,x))))
    dist = np.sum((y - x)**2)**.5
    return (dist)

number_of = 60000
number_kTest = [1,2,3,4,5,10,15,20,30,40,50]
classes = [0,1,2,3,4,5,6,7,8,9]
correct_rate = []
test_X_images = np.asarray(test_X)
test_y_labels = np.asarray(test_y)
train_X_images = np.asarray(train_X[:number_of])
train_y_labels = np.asarray(train_y[:number_of])
###########check n columns############
for k_ in range (len(number_kTest)):
    k = number_kTest[k_]
    cont = 0
    dist = []
    k_min = []
    #numbers_of_neig = []
    counter_ = [0,0,0,0,0,0,0,0,0,0]
    for ttest in range (len(test_X_images)):
        
        for test in range (len (train_X_images)):

            dist.append(calculate_dist(test_X_images[ttest],train_X_images[test]))


        dist_sorted = np.sort(dist)

        for range_k in range (k):      
            index = dist.index(dist_sorted[k])
            #numbers_of_neig.append(train_y_labels[index])

            for cc in range (len(classes)):
                if train_y_labels[index] == classes[cc]:
                    counter_[cc] = counter_[cc] + 1

        winner = np.argmax(counter_)     
        #print (counter_, winner)      
        '''most_common = max(k_min, key = k_min.count)
        
        print (k_min,most_common)

        inde = list.index(dist,most_common)
        #index = np.ndarray.argmin(np.array(most_common))
        
        nume = train_y_labels[inde]
        
        plt.imshow(test_X_images[ttest])
        plt.show()'''
        nume = winner
        if (nume == test_y_labels[ttest]):
            cont = cont + 1
        dist = []
        k_min = []
        #numbers_of_neig = []
        counter_ = [0,0,0,0,0,0,0,0,0,0]
    correct_rate.append((cont/len(test_X_images))*100)

print (correct_rate)