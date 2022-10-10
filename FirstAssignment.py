import ast
from importlib.resources import contents
from tkinter.tix import COLUMN
from tokenize import Double
import numpy as np
import pandas as pd
import csv
import os
if os.path.exists("Weatherdataset.csv"):
  os.remove("Weatherdataset.csv")
read_file= pd.read_csv('Weatherdataset.txt', sep=" ", usecols= ['Outlook','Temperature','Humidity','Windy','Play'])
read_file.to_csv (r'C:\Users\giova\Desktop\`Machine learning\Projects\Machine-learning-for-robotics-projects\Weatherdataset.csv', index=None)
wd = pd.read_csv('Weatherdataset.csv')
wd.head()
number_attributes = (wd.shape[1] - 1 )#shape [righe][colonne]
number_set = wd.shape[0]
print("\nSome line of dataset are: ")
print(wd.head())
print ("\n")
attributes = {"Outlook": {'overcast':None, 'rainy':None, 'sunny':None}, "Temperature":{'hot':None, 'cool':None, 'mild':None}, "Humidity":{'high':None, 'normal':None}, "Windy":{'FALSE':None, 'TRUE':None}}
listed_attributes=  list (attributes) #outlook temperature humidity
##########################################
print("\nStarting learning:") 
px = wd.mean(axis = 0)
PxT = px[len(px) -1 ] #prior probability of classe true
PxF = 1- PxT #prior probability of classe false)
for attribute in range (number_attributes):
  listed_attribute = list(attributes [listed_attributes[attribute]])
  for attribute_member in range(len(listed_attribute)):
    if ((listed_attribute[attribute_member] == "TRUE") or (listed_attribute[attribute_member] == "FALSE" )): #is a boolean
      #pr = (wd.query('Play')[listed_attributes[attribute]]).mean() #pr1 is P(X|y)
      pr2 =(wd.query (f'{listed_attributes[attribute]}')['Play']).mean() #skipna = true (default)
      # pr2 is P(y|X)  X={overcast, rainy, sunny} escono 3 di queste probabilità e le managio
      #So basically, P(y|X) here means, the probability of “Not playing golf” given that the weather conditions are “Rainy outlook”, “Temperature is hot”, “high humidity” and “no wind”.
    else: #is not boolean
      pr2 =(wd.query (f'{listed_attributes[attribute]} == "{listed_attribute[attribute_member]}"')['Play']).mean()
      #pr = (wd.query('Play')[listed_attributes[attribute]] == listed_attribute[attribute_member]).mean() 
      #print(f'{listed_attributes[attribute]} == {listed_attribute[attribute_member]}')
    attributes[listed_attributes[attribute]][listed_attribute[attribute_member]] = pr2 #pr2 = (pr * py)/px   #p(X|y)P(y)/P(x)
print ("\nDict of probabilities: ")
print (attributes)
print ("\n")
##########################################
print ("\nStarting input and comparing:\n")
list_in_input=[]
array_for_prod_T = []
array_for_prod_F = []
flag = 0
while (1):
  for insert in range (number_attributes): # 0, 1, 2, 3
    digited = input("insert attribute, q for quit" + listed_attributes[insert]) #1:outlook 2:temperature 3:humidity in a cicle
    list_in_input.append(digited) #filling the history of the input
    PxYT = attributes[listed_attributes[insert]][digited] #[name_sub_dictionary][element]
    PXiYF = 1 - attributes[listed_attributes[insert]][digited]
    print(PxYT)
    print (PXiYF)
    array_for_prod_T.append(PxYT)
    array_for_prod_F.append(PXiYF)
    if (digited == "q"):
      break

  print ("You have inserted these attribute: ")
  print (list_in_input)
  print ("\n")
  
  gxT= PxT * np.prod(array_for_prod_T)
  gxF = PxF * np.prod(array_for_prod_F)
  print(gxT, gxF)
 
  max = np.maximum(gxT,gxF)
  
  if max ==gxT:
    print (str("Play") + str(":") + True)
  else:
    print (str("Play") + str(":") + False)

  array_for_prod_T = [] #reset of the array
  array_for_prod_F = [] #reset of the array

  list_in_input = [] #reset of the array
  digited = input("insert attribute, q (second time) for quit")
  
  if (digited == "q"):
    break
  flag = 0


