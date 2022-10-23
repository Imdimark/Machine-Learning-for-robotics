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
file_tipe,training_file,test_file = sys.argv
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

listed_element_attributes = list(attributes[listed_attributes[0]])
if "True" in listed_element_attributes:
    print ("pipo")