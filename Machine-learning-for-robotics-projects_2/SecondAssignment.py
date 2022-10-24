import ast
from http.client import FOUND
from importlib.resources import contents
from statistics import mean
from tabnanny import check
from time import sleep
from tkinter.tix import COLUMN
from tokenize import Double
import numpy as np
import pandas as pd
import csv
import os
import sys
import matplotlib.pyplot as plt
col = 1
number_subdataset = 9

print ("Hello!")
car_dataset= pd.read_csv("mtcarsdata-4features.csv") 
turkish_dataset= pd.read_csv("turkish-se-SP500vsMSCI.csv") 
number_attributes_car_dataset = (car_dataset.shape[1] - 1 )#shape [righe][colonne]
number_set_car_dataset = car_dataset.shape[0]
number_attributes_turkish_dataset = (turkish_dataset.shape[1] - 1 )#shape [righe][colonne]
number_set_turkish_dataset = turkish_dataset.shape[0]
##############2.1#########################
xArray = list(turkish_dataset[turkish_dataset.columns[0]])
tArray = list(turkish_dataset[turkish_dataset.columns[1]])
w = (np.sum(tArray)*np.sum(xArray))/pow(2,np.sum(xArray))
x = np.array(xArray)
y = np.array(tArray)
Y = w*x
plt.scatter(x, y, color = "m",marker = "o", s = 30)
plt.plot(x, Y, color = "g")
plt.xlabel('SP')
plt.ylabel('MSCI')
##############2.2##########################
figure, axis = plt.subplots(9, 1)
figure.suptitle('10 sublots')
for ns in range (number_subdataset):
  random_subset = turkish_dataset.sample(frac = .1, ignore_index =True)
  xArray = list(random_subset[random_subset.columns[0]])
  tArray = list(random_subset[random_subset.columns[1]])
  w = (np.sum(tArray)*np.sum(xArray))/pow(2,np.sum(xArray))
  x = np.array(xArray,dtype='float64')
  y = np.array(tArray,dtype='float64')
  Y = w*x
  axis[ns].scatter(x, y, color = "m",marker = "o", s = 30)
  axis[ns].plot(x, Y, color = "g")
  plt.xlabel('SP')
  plt.ylabel('MSCI')
plt.show()

############2.3####################
xArray = list(car_dataset[car_dataset.columns[1]])
tArray = list(car_dataset[car_dataset.columns[4]])
#w = (np.sum(tArray)*np.sum(xArray))/pow(2,np.sum(xArray))

x = np.array(xArray)
y = np.array(tArray)

x_ = np.mean(x)
y_ = np.mean(y)


w1 = np.sum((x - x_)*(y - y_)) / np.sum(pow(2, (x - x_)) )
w0 = y_ - (w1 * x_) 


Y = w0 + (x *w1)
print (Y)
plt.scatter(x, y, color = "m",marker = "o", s = 30)
plt.plot(x, Y, color = "g")
plt.xlabel('MPG')
plt.ylabel('Weight')
plt.show()




##########2.4##############


cols = [0,2,3,4]
xArray = car_dataset [car_dataset.columns[cols]]
xArray = np.array(xArray)
#print (xArray)
#print(len(xArray))


ceckArray = []
for qq in range (len(xArray)):
  if xArray[qq][0] not in ceckArray:
    ceckArray.append (xArray[qq][0])

for qq in range (len(xArray)):
  xArray[qq][0] = ceckArray.index(xArray[qq][0])

xArray = np.array(xArray,dtype='float64')
yArray = np.array(car_dataset[car_dataset.columns[1]],dtype='float64')

print (yArray)

q = np.linalg.pinv(xArray)

print (q)
print("weeeeeeeeee")
w = q * yArray
print (w)


w = np.dot(q,yArray)
print (w)





