from array import array
import ast
import collections
from http.client import FOUND
from importlib.resources import contents
from statistics import mean
from tabnanny import check
from time import sleep
from tkinter.tix import COLUMN
from tokenize import Double
from turtle import color
import numpy as np
import pandas as pd
import csv
import random
import os
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

col = 1
number_subdataset = 9

def two_one (turkish_dataset,plot):
  
  xArray = list(turkish_dataset[turkish_dataset.columns[0]])
  tArray = list(turkish_dataset[turkish_dataset.columns[1]])
  den = np.power (xArray,2)
  w = (np.dot(np.sum(tArray),np.sum(xArray)))/np.sum(den)
  x = np.array(xArray)
  y = np.array(tArray)
  Y = np.dot(x,w)
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
    den = np.power (xArray,2)
    w = (np.dot(np.sum(tArray),np.sum(xArray)))/np.sum(den)
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
  w0 = np.subtract (y_ , np.dot(w1, x_) )
  Y = np.add (w0 , np.dot(x,w1))
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
  xArray = np.array(xArray,dtype='float64')
  tArray = np.array(list(car_dataset[car_dataset.columns[1]]),dtype='float64')
  xArray_sword = np.linalg.pinv(xArray)
  w = np.dot (xArray_sword ,tArray)
  w = np.array(w,dtype='float64' )
  xArray = np.array(xArray,dtype='float64')
  Y = np.dot (xArray, w)
#####################################
  norm_xw = np.power(np.linalg.norm(np.dot(xArray,w)),2)
  Jmse = norm_xw/ len(xArray)
############################################
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
  return (tArray, Y, Jmse)
def calculate_Jmse(tArray, Y, N):
  #if dimension == "mono":
  Jmse = np.square(np.subtract(Y,tArray)).mean()
  return (Jmse)
def random_splitting(dataset):
  five = dataset.sample(frac=0.05)
  joined_df = dataset.merge(five, how="outer", left_index=True, right_index=True)
  joined_df.iloc[:,-1].isna()
  ninetyfive = dataset[joined_df.iloc[:,-1].isna()]
  return (five, ninetyfive)

car_dataset= pd.read_csv("mtcarsdata-4features.csv") 
turkish_dataset= pd.read_csv("turkish-se-SP500vsMSCI.csv") 
number_attributes_car_dataset = (car_dataset.shape[1] - 1 )#shape [righe][colonne]
number_set_car_dataset = car_dataset.shape[0]
number_attributes_turkish_dataset = (turkish_dataset.shape[1] - 1 )#shape [righe][colonne]
number_set_turkish_dataset = turkish_dataset.shape[0]
################ 2 ################
print ("Hello!")
two_one(turkish_dataset,True)
two_two(turkish_dataset)
two_three(car_dataset,True)
two_for(car_dataset,True)

################ 3 ################
p = 9
Jmse1 = np.empty((p,2))
Jmse2 = np.empty((p,2))
Jmse3 = np.empty((p,2))

for repeat in range (p):
  turkish_dataset_05,turkish_dataset_95 = random_splitting(turkish_dataset)
  car_dataset_05,car_dataset_95 = random_splitting(car_dataset)
  tArray, Y = two_one(turkish_dataset_05,False)
  Jmse1 [repeat][0] = calculate_Jmse(tArray, Y, len(turkish_dataset_05))
  tArray, Y = two_one(turkish_dataset_95,False)
  Jmse1[repeat][1] = calculate_Jmse(tArray, Y, len(turkish_dataset_95))
  tArray, Y =two_three(car_dataset_05,False)
  Jmse2 [repeat][0] = calculate_Jmse(tArray, Y, len(car_dataset_05))
  tArray, Y =two_three(car_dataset_95,False)
  Jmse2 [repeat][1] = calculate_Jmse(tArray, Y, len(car_dataset_95))
  tArray, Y, Jmse =two_for(car_dataset_05,False)
  Jmse3[repeat][0] =Jmse
  tArray, Y,Jmse=two_for(car_dataset_95,False)
  Jmse3[repeat][1] =Jmse
figure, axis = plt.subplots(3, 1)
figure.suptitle('Jmse on 5 perc (red) and 95 perc (lime) of the dataset')

colors = ['red','lime']
axis[0].hist(Jmse1, color=colors)
axis[0].set_xlabel ('Jmse')
axis[0].set_ylabel ('Number of occurrences')

axis[1].hist(Jmse2, color=colors)
axis[1].set_xlabel ('Jmse')
axis[1].set_ylabel ('Number of occurrences')

axis[2].hist(Jmse3, color=colors)
axis[2].set_xlabel ('Jmse')
axis[2].set_ylabel ('Number of occurrences')
plt.show()


