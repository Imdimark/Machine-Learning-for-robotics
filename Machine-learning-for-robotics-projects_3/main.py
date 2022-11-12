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
import sys
import datetime

def additional_point(confusion_matrix):
    array = np.zeros(6)
    n_correct = np.trace(confusion_matrix)    
    accuracy = n_correct/len(test_X_images) # is equal to correct rate calculated in point 3

    correct_0 = confusion_matrix[0][0]   # 1 if it is the number, 0 if not
    false_0 = confusion_matrix[0][1]
    correct_1 = confusion_matrix[1][0]   
    false_1 = confusion_matrix[1][1]
    sensitivity = correct_1/(correct_1 + false_0)     #probability to correctly recognize class 1
    specifiticy = correct_0/(correct_0 + false_1)
    precision = correct_1/(correct_1 + false_1)
    recall = sensitivity
    F_measure = 2 * ( (precision *recall)/(precision + recall) )
    array[0] = (sensitivity)
    array[1] = (specifiticy)
    array[2] = (precision)
    array[3] = (recall)
    array[4] = (F_measure)
    array[5] = (accuracy)   
    return array

def calculate_dist(x,y): 
    #sum = 0
    #for X in range (len(x)):
    #    for Y in range (len(y)):
    #        sum = sum + np.square(x[X][Y] - y[X][Y])  #too slow
    #return (np.sqrt(sum))
    #return (np.linalg.norm(x - y, ord=2))
    #return np.sqrt(np.sum((x - y) ** 2))   
    x = x.flatten() #convert in 1D
    y = y.flatten()
    dist = np.sum((y - x)**2)**.5 #euclidean distance
    return (dist)

def Classifier (test_X_images, test_y_labels, train_X_images, train_y_labels, k, classes): #if the number of args is < 5 then the execution will be interrupted
    cont = 0
    dist = []
    winner_array =[]
    confusion_matrix = np.zeros((len(classes), len(classes)))
    #numbers_of_neig = []
    counter_ = [0,0,0,0,0,0,0,0,0,0] #memorizes the occourrences of single classe
    for ttest in range (len(test_X_images)):       
        for test in range (len (train_X_images)):
            dist.append(calculate_dist(test_X_images[ttest],train_X_images[test]))
        dist_sorted = np.sort(dist)
        for range_k in range (k):      
            index = dist.index(dist_sorted[range_k])
            for cc in range (len(classes)):
                if train_y_labels[index] == classes[cc]:
                    counter_[cc] = counter_[cc] + 1       
        winner = np.argmax(counter_)  # in case of two or more max with the same values takes the first encountered
        winner_array.append(winner) #this memorize the classification for the courrent k 
        confusion_matrix[test_y_labels[ttest]] [winner] += 1    
        if (winner == test_y_labels[ttest]):
            cont = cont + 1 #to optimize the space i prefere to mantain an integer instead a one more column (1x10000)        
        dist = [] #clean the memory
        counter_ = [0,0,0,0,0,0,0,0,0,0]
    denominatore = len(test_X_images)
    correct_rate= (cont/denominatore)*100 
    return (winner_array, correct_rate, confusion_matrix)
############# init #############
start = datetime.datetime.now()
print ("starting time: " + str (start))
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = tf.keras.utils.normalize(train_X, axis = 1) #optimization
test_X = tf.keras.utils.normalize(test_X, axis = 1)
number_of = 6000
#nargin_min = 5
number_kTest = [1,2,3,4,5,10,15,20,30,40,50]
#number_kTest = [1,2,3]
classes = [0,1,2,3,4,5,6,7,8,9]
correct_rate = [] #memorizes the various correct rate 
classification_array = []
test_X_images = np.asarray(test_X) 
test_y_labels = np.asarray(test_y)
train_X_images = np.asarray(train_X[:number_of])
train_y_labels = np.asarray(train_y[:number_of])

############# checks #############
if np.max(number_kTest) >= number_of and np.max(number_kTest) > 0: #check  0 < k <= number of images used from training set
    print ("error in k number")
    sys.exit()
if len(test_X_images[1])!= len(train_X_images[1]): #check taht training set e test set has the same column
    print ("error in number of columns")
    sys.exit()
############# point 2 #############
correct_rate_array = []
for k_ in range (len(number_kTest)):    
    k = number_kTest[k_] #k that i'm using in this cycle
    print ("Calculation under this K: " + str(k))
    winner_array,correct_rate= Classifier (test_X_images, test_y_labels, train_X_images, train_y_labels, k, classes)
    classification_array.append(winner_array)
    correct_rate_array.append (correct_rate)
duration = datetime.datetime.now() - start
print ("duration of the point 2:" + str(duration))
#stamp = input ('Do you want to print the array with the test set classification for all the K? Y/N')
print ("Next step: a graph with the errors rate will be shown")
#if (stamp == "Y"):
#    print (winner_array)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(number_kTest, correct_rate_array)
ax.set_xlabel('number of Nearest Neighbors (k)')
ax.set_ylabel('Accuracy (%)')
ax.set_title("error rate depending on k")
plt.show()

############# point 3 #############
correct_rate = [] #clean the array
correct_rate_array = []
old_classes = classes
classes = [0,1] #now they are 2, 0 and 1, 1 if it is the digit
start = datetime.datetime.now()
print ("starting time 3 point: " + str (start))
fig, ax = plt.subplots(10,1)

for t in range (len(old_classes)):
    print ("Number: " + str(t) ) 
    test_y_labels_2classes = []
    train_y_labels_2classes = []
    number_1 = 0
    print(test_y_labels)
    
    for i in range (len (test_X_images)):#here converts the number taken in consideration as 1 and the othe 9 as 0
        if test_y_labels[i] == t:
            test_y_labels_2classes.append(1) 
            number_1 += 1  
        else:
            test_y_labels_2classes.append(0)
    for i in range (len (train_X_images)):    
        if train_y_labels [i] == t:
            train_y_labels_2classes.append(1)  
        else:
            train_y_labels_2classes.append(0)

    for k_ in range (len(number_kTest)):    
        k = number_kTest[k_] #k that i'm using in this cycle
        print ("Calculation under this K: " + str(k))
        winner_array,correct_rate = Classifier (test_X_images, test_y_labels_2classes, train_X_images, train_y_labels_2classes, k, classes)
        correct_rate_array.append(correct_rate)

        
    ax[t].plot(number_kTest, correct_rate_array)
    ax[t].set_title ('Error rate depending on various K, of the number: '+ str(t))
    ax[t].set_xlabel('number of Nearest Neighbors (k)')
    ax[t].set_ylabel('Accuracy (%)')
    correct_rate_array = []
duration = datetime.datetime.now() - start
print ("duration of the point 3 :" + str(duration))
plt.show()

############# additional point #############
start = datetime.datetime.now()
print ("starting time additional point: " + str (start))
correct_rate = [] #clean the array
correct_rate_array = []
classes = [0,1] #now they are 2, 0 and 1, 1 if it is the digit
table = [0 for x in range(len(number_kTest)* len(old_classes))]
c = 0
#table = [[0 for i in range(10)] for j in range(10)]
#table = np.array((10,10))
for t in range (len(old_classes)):
    print ("Number: " + str(t) ) 
    test_y_labels_2classes = []
    train_y_labels_2classes = []
    number_1 = 0

    for i in range (len (test_X_images)): #here converts the number taken in consideration as 1 and the othe 9 as 0
        if test_y_labels[i] == t:
            test_y_labels_2classes.append(1) 
            number_1 += 1  
        else:
            test_y_labels_2classes.append(0)
    for i in range (len (train_X_images)):    
        if train_y_labels [i] == t:
            train_y_labels_2classes.append(1)  
        else:
            train_y_labels_2classes.append(0)

    for k_ in range (len(number_kTest)):
        k = number_kTest[k_] #k that i'm using in this cycle
        print ("Calculation under this K: " + str(k))
        pluto = np.zeros((10,6))
        for sample in range(10):
            indexs = np.random.randint (1, 10000, 1000)
            test_X_images_sampled = test_X_images [indexs]
            test_y_labels_sampled = test_y_labels [indexs]
            winner_array,correct_rate,confusion_matrix = Classifier (test_X_images_sampled, test_y_labels_2classes, train_X_images, train_y_labels_2classes, k, classes)
            pluto[sample] = additional_point(confusion_matrix) 
        standard_deviation = np.zeros(6)
        averages = [0,0,0,0,0,0,0,0]
        for x in range (6):
            standard_deviation[x] = np.std(pluto[:,x]) 
            averages[0] = t
            averages [1] = k             
            averages[x+2] = ( str(np.mean(pluto[:,x])) + ('(') + str(np.std(pluto[:,x]) )  + (')') )  # sensitivity specificity precision recall f_measure recall
        table[c ] = averages   # row = k, column = numbers
        c = c + 1
        
df = pd.DataFrame(table, columns = ("number", "k_number" ,"sensitivity", "specificity", "precision", "recall", "f_measure", "recall"))
df.to_csv("./Table.csv", sep=',',index=False)
print (df)
duration = datetime.datetime.now() - start
print ("duration of the point 3 :" + str(duration))
