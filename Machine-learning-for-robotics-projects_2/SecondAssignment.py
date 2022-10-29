from array import array
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
from mpl_toolkits.mplot3d import Axes3D
col = 1
number_subdataset = 9


def two_one (turkish_dataset,plot):
  
  xArray = list(turkish_dataset[turkish_dataset.columns[0]])
  tArray = list(turkish_dataset[turkish_dataset.columns[1]])
  w = (np.sum(tArray)*np.sum(xArray))/pow(2,np.sum(xArray))
  x = np.array(xArray)
  y = np.array(tArray)
  Y = w*x
  if plot:
    plt.title("One-dimensional problem without intercept on the Turkish stock exchange data")
    plt.scatter(x, y, color = "m",marker = "o", s = 30)
    plt.plot(x, Y, color = "g")
    plt.xlabel('SP')
    plt.ylabel('MSCI')
    plt.show()
  return (y, Y)
def two_two (turkish_dataset):
  figure, axis = plt.subplots(9, 1)
  figure.suptitle('10 random subset (10%)')
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
    axis[ns].set_xlabel ('SP')
    axis[ns].set_ylabel ('MSCI')
    plt.xlabel('SP')
    plt.ylabel('MSCI')
  plt.show()
  return (tArray, Y)
def two_three(car_dataset, plot):
  xArray = list(car_dataset[car_dataset.columns[1]])
  tArray = list(car_dataset[car_dataset.columns[4]])
  x = np.array(xArray)
  y = np.array(tArray)
  x_ = np.mean(x)
  y_ = np.mean(y)
  w1 = np.sum((x - x_)*(y - y_)) / np.sum(pow(2, (x - x_)) )
  w0 = y_ - (w1 * x_) 
  Y = w0 + (x *w1)
  if plot:
    plt.title("One-dimensional problem with intercept on the Motor Trends car data")
    plt.scatter(x, y, color = "m",marker = "o", s = 30)
    plt.plot(x, Y, color = "g")
    plt.xlabel('MPG')
    plt.ylabel('Weight')
    plt.show()
  return (y, Y)
def two_for(car_dataset,plot):
  cols = [2,3,4]
  xArray = car_dataset [car_dataset.columns[cols]]
  #xArray = xArray.to_numpy()
  #xArray = np.transpose(xArray)
  xArray = np.array(xArray,dtype='float64')
  tArray = np.array(list(car_dataset[car_dataset.columns[1]]),dtype='float64')
  #tArray = np.transpose(tArray)
  #xs = np.linalg.inv(np.dot(np.transpose(xArray),xArray))
  #xArray_sword = np.dot(xs,np.transpose(xArray))
  xArray_sword = np.linalg.pinv(xArray)
  w = np.dot (xArray_sword ,tArray)
  w = np.array(w,dtype='float64' )
  xArray = np.array(xArray,dtype='float64')
  Y = np.dot (xArray, w)
  if plot:
    plt.rcParams['legend.fontsize'] = 12
    fig = plt.figure()
    ax = fig.add_subplot(projection ='3d')
    ax.scatter(xArray[:, 0], xArray[:, 1], xArray[:, 2], tArray, label ='y', s = 5, color="red")
    ax.scatter(xArray[:, 0], xArray[:, 1], xArray[:, 2], Y, label ='regression', s = 5, color="blue")
    ax.set_title("Multi-dimensional problem on the complete MTcars data")
    ax.legend()
    ax.set_xlabel('disp', fontsize=10)
    ax.set_ylabel('hp', fontsize=10)
    ax.set_zlabel('weight', fontsize=10)
    ax.view_init(45, 0)
    plt.show()
  return (tArray, Y)
def calculate_Jmse(tArray, Y, N):
  pluto = pow (2, (tArray - Y))
  Jmse = (np.sum (pluto))/N
  return (Jmse)
car_dataset= pd.read_csv("mtcarsdata-4features.csv") 
turkish_dataset= pd.read_csv("turkish-se-SP500vsMSCI.csv") 
number_attributes_car_dataset = (car_dataset.shape[1] - 1 )#shape [righe][colonne]
number_set_car_dataset = car_dataset.shape[0]
number_attributes_turkish_dataset = (turkish_dataset.shape[1] - 1 )#shape [righe][colonne]
number_set_turkish_dataset = turkish_dataset.shape[0]
################ 2 ################
two_one(turkish_dataset,True)
two_two(turkish_dataset)
two_three(car_dataset,True)
two_for(car_dataset,True)

################ 3 ################
p = 90
Jmse1 = np.empty((p,2))
Jmse2 = np.empty((p,2))
Jmse3 = np.empty((p,2))

for repeat in range (p):
  
  turkish_dataset_05 = turkish_dataset.sample(frac = .05, ignore_index =True)
  turkish_dataset.index.difference(turkish_dataset_05.index)
  turkish_dataset_95 = turkish_dataset.iloc[turkish_dataset.index.difference(turkish_dataset_05.index)]

  car_dataset_05 = car_dataset.sample(frac = .05, ignore_index =True)
  car_dataset.index.difference(car_dataset_05.index)
  car_dataset_95 = car_dataset.iloc[car_dataset.index.difference(car_dataset_05.index)]
  
  tArray, Y = two_one(turkish_dataset_05,False)
  Jmse1 [repeat][0] = calculate_Jmse(tArray, Y, len(turkish_dataset_05))
  tArray, Y = two_one(turkish_dataset_95,False)
  Jmse1[repeat][1] = calculate_Jmse(tArray, Y, len(turkish_dataset_95))
  
  
  tArray, Y =two_three(car_dataset_05,False)
  Jmse2 [repeat][0] = calculate_Jmse(tArray, Y, len(car_dataset_05))
  tArray, Y =two_three(car_dataset_95,False)
  Jmse2 [repeat][1] = calculate_Jmse(tArray, Y, len(car_dataset_95))
  
  tArray, Y =two_for(car_dataset_05,False)
  Jmse3[repeat][0] =calculate_Jmse(tArray, Y, len(car_dataset_05))
  tArray, Y =two_for(car_dataset_95,False)
  Jmse3[repeat][1] =calculate_Jmse(tArray, Y, len(car_dataset_95))
  

plt.show()

