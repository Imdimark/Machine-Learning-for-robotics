import ast
from importlib.resources import contents
from tkinter.tix import COLUMN
from tokenize import Double
import numpy as np
import pandas as pd
import csv
import os
import sys
print ("Hello!")
file_tipe,training_file,test_file = sys.argv
print (file_tipe, training_file, test_file)
if os.path.exists("ConToCsv.csv"):
  os.remove("ConToCsv.csv")
read_file= pd.read_csv(sys.argv[1], sep=" ")
read_file.to_csv (r'C:\Users\giova\Desktop\`Machine learning\Projects\Machine-learning-for-robotics-projects\ConToCsv.csv', index=None)
wd = pd.read_csv("ConToCsv.csv")
print("\nStarting learning: \n")
wd.head()
number_attributes = (wd.shape[1] - 1 )#shape [righe][colonne]
number_set = wd.shape[0]
attributes = {}
PY = (list(wd[wd.columns[wd.shape[1]-1]]).count(True))/number_set #prior probability of class True #### 0.6 qualcosa
PN = (list(wd[wd.columns[wd.shape[1]-1]]).count(False))/number_set#prior probability of class False
for aa in range (0,number_attributes,1): 
  attributes[wd.columns[aa]] = {}
  for ab in range (len(list(set(wd[wd.columns[aa]])))): #ab vara nel range dei singoli, interni ad aa colonna. ss sarebbe as esempio overcast,rainy,sunny
    ss =(list(set(wd[wd.columns[aa]])))[ab]
    counter = 0
    for CheckEqualTrue in range (number_set):
      if list(wd[wd.columns[aa]])[CheckEqualTrue] == ss: 
        if list(wd[wd.columns[wd.shape[1]-1]])[CheckEqualTrue] == True:
          counter = counter + 1
    Pxty = counter/((list(wd[wd.columns[wd.shape[1]-1]]).count(True))) #Pxtf = 1-Pxty
    attributes[wd.columns[aa]][ss] = Pxty
print(attributes)
listed_attributes= list (attributes) #outlook temperature humidity
print ("Learning phase ended, starting classification with the test set \n")
array_for_prod_T = [] #reset of the array
array_for_prod_F = [] #reset of the array
targetClass_row = []
ts = pd.read_csv(sys.argv[2])
print (sys.argv[2])
ts.head()
number_attributes_ts = (ts.shape[1])#shape [righe][colonne]
number_set_ts = ts.shape[0]
for q in range (0, number_set_ts, 1): #for every row of the test set 1 10
  for insert in range (number_attributes_ts-1): # 0, 1, 2, 3
    #digited = input("insert attribute, q for quit" + listed_attributes[insert]) #1:outlook 2:temperature 3:humidity in a cicle
    #list_in_input.append(digited) #filling the history of the input
    PxYT = attributes[listed_attributes[insert]][ts.iat[q,insert]] #[name_sub_dictionary][element]        
    PXiYF = 1 - PxYT
    print (PxYT ,PXiYF)
    if (PxYT != 0):
      array_for_prod_T.append(PxYT)
    if (PXiYF != 0):
      array_for_prod_F.append(PXiYF)   
  gxT= PY * np.prod(array_for_prod_T)
  gxF = PN * np.prod(array_for_prod_F)
  #print (gxT, gxF)
  max = np.maximum(gxT,gxF)
  if max ==gxT:
    targetClass_row.append(True)
  else:
    targetClass_row.append(False)
  array_for_prod_T = [] #reset of the array
  array_for_prod_F = [] #reset of the array
print("Target class row: ")
print (targetClass_row)

if (number_attributes_ts == number_attributes + 1): # target class is present
  for err in range (number_set_ts):
    if targetClass_row[err] == list(wd[wd.columns[number_attributes_ts]])[err] :
      NumbErr = NumbErr + 1
  print (NumbErr)
  ErrorRate = NumbErr / number_set_ts
  print ("error rate" + ErrorRate)

ts.insert(number_attributes_ts,list(wd)[number_attributes], targetClass_row ) #add 1 row on the right
print ("\n Result:\n" )
print (ts)
