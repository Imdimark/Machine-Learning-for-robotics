import ast
from http.client import FOUND
from importlib.resources import contents
from tkinter.tix import COLUMN
from tokenize import Double
import numpy as np
import pandas as pd
import csv
import os
import sys
a = 1

print ("Hello!")
file_tipe,training_file = sys.argv
if os.path.exists("ConToCsv.csv"):
  os.remove("ConToCsv.csv")
read_file= pd.read_csv(sys.argv[1], sep=" ")
read_file.to_csv (r'C:\Users\giova\Desktop\`Machine learning\Projects\Machine-learning-for-robotics-projects\ConToCsv.csv', index=None)
EntirePandasDataset= pd.read_csv("ConToCsv.csv") 

wd = EntirePandasDataset.sample(n = 10, ignore_index =True) # training set with 10 randomly chosen patterns
ts = EntirePandasDataset.sample(n = 4, ignore_index =True) # test set with the remaining 4 random patterns

print("\nStarting learning: \n")
wd.head()
number_attributes = (wd.shape[1] - 1 )#shape [righe][colonne]
number_set = wd.shape[0]
attributes = {} 
attributes_v = {}  #it contains the 1/v value only
PY = (list(wd[wd.columns[wd.shape[1]-1]]).count(True))/number_set #prior probability of class True #### 0.6 qualcosa
PN = (list(wd[wd.columns[wd.shape[1]-1]]).count(False))/number_set#prior probability of class False
for aa in range (0,number_attributes,1): 
  attributes[wd.columns[aa]] = {}
  attributes_v[wd.columns[aa]] = {}
  for ab in range (len(list(set(wd[wd.columns[aa]])))): #ab vara nel range dei singoli, interni ad aa colonna. ss sarebbe as esempio overcast,rainy,sunny
    ss =(list(set(wd[wd.columns[aa]])))[ab]
    counter = 0
    for CheckEqualTrue in range (number_set):
      if list(wd[wd.columns[aa]])[CheckEqualTrue] == ss: 
        if list(wd[wd.columns[wd.shape[1]-1]])[CheckEqualTrue] == True:
          counter = counter + 1
#################Laplace smoother###############            
    v = len(list(set(wd[wd.columns[aa]])))
    attributes_v[wd.columns[aa]][ss] = v
    Pxty =(counter + a) / (((list(wd[wd.columns[wd.shape[1]-1]]).count(True))) + (a*v))  #Pxtf = 1-Pxty
    attributes[wd.columns[aa]][ss] = Pxty
listed_attributes= list (attributes) #outlook temperature humidity
print ("Learning phase ended, starting classification with the test set \n")
array_for_prod_T = [] #reset of the array
array_for_prod_F = [] #reset of the array
targetClass_row = []
ts.head()
number_attributes_ts = (ts.shape[1])#shape [righe][colonne]
print(number_attributes_ts)
number_set_ts = ts.shape[0]
print (attributes)
for q in range (0, number_set_ts, 1): #for every row of the test set 1 10
  for insert in range (number_attributes_ts-1): # 0, 1, 2, 3
    # se viene trovato dentro il set 
    listed_element_attributes = list(attributes[listed_attributes[insert]])
    if ts.iat[q,insert] in listed_element_attributes:
      PxYT = attributes[listed_attributes[insert]][ts.iat[q,insert]] #[name_sub_dictionary][element]        
      PXiYF = 1 - PxYT
    else: ## Adjust if the test_set's vale is not present in the training_set
      PxYT = 1/attributes_v[listed_attributes[insert]][wd.iat[q,insert]]
      PXiYF = 1 - PxYT  
    array_for_prod_T.append(PxYT)
    array_for_prod_F.append(PXiYF)   
  gxT= PY * np.prod(array_for_prod_T)
  gxF = PN * np.prod(array_for_prod_F)
  max = np.maximum(gxT,gxF)
  if max == gxT:
    targetClass_row.append(True)
  elif max == gxF:
    targetClass_row.append(False)
  array_for_prod_T = [] #reset of the array
  array_for_prod_F = [] #reset of the array
print (targetClass_row)
NumbErr = 0
if (number_attributes_ts == number_attributes + 1): # target class is present
  for err in range (number_set_ts):
    if targetClass_row[err] == list(wd[wd.columns[number_attributes_ts-1]])[err] :
      NumbErr = NumbErr + 1
  ErrorRate = NumbErr / number_set_ts
ts = ts.iloc[: , :-1]
ts.insert((number_attributes_ts-1), list(wd)[number_attributes],targetClass_row )
print ("\n Result:\n" )
print (ts)
print ("Error rate: ")
print (ErrorRate)
