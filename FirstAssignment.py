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
#print(wd.head())

attributes = {"Outlook": {'overcast', 'rainy', 'sunny'}, "Temperature":{'hot', 'cool', 'mild'}, "Humidity":{'high', 'normal'}, "Windy":{'FALSE', 'TRUE'}}
listed_attributes=  list (attributes) #outlook temperature humidity
for attribute in range (number_attributes):
  listed_attribute = list(attributes [listed_attributes[attribute]])

  for attribute_member in range(len(listed_attribute)):
    #print(listed_attribute[attribute_member])
    #print( (wd.query('Play')[listed_attributes[attribute]]== listed_attribute[attribute_member]).mean() )
    print(listed_attributes[attribute] + "confrontato con" + listed_attribute[attribute_member])
    pippo = (wd.query('Play')[listed_attributes[attribute]]== listed_attribute[attribute_member]).mean()
    #attributes[listed_attributes[attribute]][listed_attribute[attribute_member]] = pippo
list_in_input=[]
probability = 0
while (1):
  for insert in range (number_attributes):
    digited = input("insert attribute, q for quit" + listed_attributes[insert])
    list_in_input.append(digited)
    probability += attributes
    print (probability)
    if (digited == "q"):
      exit
  print (list_in_input)
  print (probability)
  if (input == "q"):
    exit



"""print( (wd.query('Play')[Attributes[0]]=='overcast').mean() )
print( (wd.query('Play')[Attributes[0]]=='rainy').mean() )
print( (wd.query('Play')[Attributes[0]]=='sunny').mean() )"""

#print( (wd.query('Play')[attributes[0]]=='rainy').mean() )
#print( (wd.query('Play')['Outlook']=='rainy').mean() )





#Weatherdataset = open('Weatherdataset.csv', mode='r')
#csv_filename = 'Weatherdataset.csv'
#with open('Weatherdataset.csv', newline='') as f:
#    reader = csv.reader(f)
#    data = list(reader)
#number_set = len(data)
#attributes = 4
#Outlook = ["overcast", "rainy", "sunny"]  
#Temperature = ["hot", "cool", "mild"]
#Temperature = ["hot", "cool", "mild"]
#for y in range (0, 1, attributes):
  #for x in range(1, 1, number_set):
    #riga colonna
    #if data [x][y] ==  


"""for convert in number_set:
  if (wd.head([convert][number_attributes]) == "yes"):
    wd.head([convert][number_attributes]) = True
  
  if else:
    wd.head([convert][number_attributes]) = False"""
  
"""Outlook = {'overcast', 'rainy', 'sunny'}
Temperature = {'hot', 'cool', 'mild'}
Humidity = {'high', 'normal'}
Windy = {'FALSE', 'TRUE'}
attributes = {"Outlook": Outlook, "Temperature":Temperature, "Humidity":Humidity, "Windy":Windy}"""




