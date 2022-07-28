import os
import cv2
import pytesseract as pyt
import glob
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import csv
import seaborn as sns
# import re
# nltk.download('punkt') #sentence tokenizer puunctuation kit
# nltk.download('popular')
from nltk.tokenize import word_tokenize as wt
from string import punctuation
pun = list(punctuation)
pun.append('\n')
pyt.pytesseract.tesseract_cmd = r"D:\\Softwares\\Installed_Softwares\\Anaconda\\Library\\bin\\tesseract.exe"

def img_to_string(img_source):
    text=''
    img = cv2.imread(str(img_source))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    text = pyt.image_to_string(img) 
#     return text[1:len(text)-2]
    return text

def nonpunct(text):
    text=text.lower()
    new_text=""
  #cleaned_tokens = [token for token in lower_case if token not in stop_words and token not in pun]
    for i in text:
        # print(i)
        
        if i.isalnum:
            if str(i) not in pun:
                # print(i)
                new_text += i
            else:
                new_text +=' '
        else:
            new_text += ' '
        # for i in cleaned_tok:
    return new_text.upper()


license_plates3 = []
license_plates_text3=[]
license_plates3.append(os.listdir("D:\\AEEE_THINGS\\Sem_5\\Projects\\FDS\\Datasets\\Indian License Plates\\Indian License Plates\\Indian Plates\\positive\\"))
# license_plates_text3 = np.ndarray(shape = (len(license_plates3[0]),1), dtype = str)
# print(license_plates_text3.shape)
# print(license_plates3[0][2])


for j in range(1,len(license_plates3[0])):
#     print(i)
    # license_plates_text3[j-1]=(img_to_string("Datasets\\Indian License Plates\\Indian License Plates\\Indian Plates\\positive\\"+str(license_plates3[0][j])))
    license_plates_text3.append(img_to_string("D:\\AEEE_THINGS\\Sem_5\\Projects\\FDS\\Datasets\\Indian License Plates\\Indian License Plates\\Indian Plates\\positive\\"+str(license_plates3[0][j])))
    # print(str(j)+' : '+str(license_plates3[0][j]))
    # print("Recognised text : "+str(license_plates_text3[j-1]))

#     print(license_plates[0][i])
# print(license_plates_text)


print("Recognised text : "+str(len(license_plates_text3)))


df = pd.DataFrame(license_plates_text3)
df.to_csv('Positive Indian Number Plate.csv',index = False)

data = pd.read_csv('Positive Indian Number Plate.csv')

data.head()

# data.rename(columns={"0": "Folder : Positive Indian Number Plates "}, inplace=True)
data['Folder : Positive Indian Number Plates']=data['0']
del data['0']
data.head()
# data.shap

for i in range(1,len(data['Folder : Positive Indian Number Plates'])):
    data['Folder : Positive Indian Number Plates'].iloc[i] = nonpunct(data['Folder : Positive Indian Number Plates'].iloc[i])

    data.to_csv("omg.csv")


dataset = pd.read_csv('omg.csv')
dataset.rename(columns = {"Folder : Positive Indian Number Plates": "License-Plate"}, inplace=True)
dataset.drop(['Unnamed: 0'], axis='columns', inplace=True)
dataset.to_csv('file_name.csv', index=False)
with open('file_name.csv', encoding = 'utf-8') as input, open('new.csv', 'w', newline='') as output:
    writer = csv.writer(output)
    for row in csv.reader(input):
        if any(field.strip() for field in row):
            writer.writerow(row)
dataset = pd.read_csv('new.csv', encoding = 'unicode_escape')
dataset


State = []
for row in dataset['License-Plate']:
    if "KA" in row :   
        State.append('Karnataka')
    elif "TN" in row:   
        State.append('Tamil Nadu')
    elif "MH" in row:  
        State.append('Maharastra')
    elif "DL" in row:  
        State.append('Delhi')
    elif "WB" in row:  
        State.append('West Bengal')
    elif "PB" in row:  
        State.append('Punjab')
    elif "GJ" in row:  
        State.append('Gujarat')
    elif "UP" in row:  
        State.append('Uttar Pradesh')
    elif "AP" in row:  
        State.append('Andra Pradesh')
    elif "MP" in row:  
        State.append('Madhya Pradesh')
    elif "AR" in row:  
        State.append('Arunachal Pradesh')
    elif "AS" in row:  
        State.append('Assam')
    elif "Mn" in row:  
        State.append('Manipur')
    elif "HR" in row:  
        State.append('Haryana')
    elif "la" in row:  
        State.append('Ladakh')
    elif "AN" in row:  
        State.append('Andaman and Nicobar Islands')
    elif "KL" in row:  
        State.append('Kerala')
    elif "BR" in row:  
        State.append('Bihar')
    elif "CH" in row:  
        State.append('Chandigarh')  
    elif "CG" in row:  
        State.append('Chhattisgarh')
    elif "GA" in row:  
        State.append('Goa')
    elif "HP" in row:  
        State.append('Himachal Pradhesh')
    elif "JK" in row:  
        State.append('Jammu & Kashmir')
    elif "JH" in row:  
        State.append('Jharkhand')
    elif "LD" in row:  
        State.append('Lakshadweep')
    elif "MN" in row:  
        State.append('Manipur')
    elif "ML" in row:  
        State.append('Meghalaya')
    elif "MZ" in row:  
        State.append('Mizoram')
    elif "NL" in row:  
        State.append('Nagaland')
    elif "OD" in row:  
        State.append('Odisha')
    elif "OR" in row:  
        State.append('Odisha')
    elif "PY" in row:  
        State.append('Puducherry')
    elif "RJ" in row:  
        State.append('Rajastan')
    elif "SK" in row:  
        State.append('Sikkim')
    elif "TL" in row:  
        State.append('Telangana')
    elif "TR" in row:  
        State.append('Tripura')
    elif "UK" in row:  
        State.append('Uttarakhand')
    elif "UA" in row:  
        State.append('Uttarakhand')
    elif "DD" in row:  
        State.append('Dadra and Nagar Haveli')
    elif "DN" in row:  
        State.append('Dadra and Nagar Haveli')
    elif " " in row:  
        State.append('NA')
    else:           
        State.append('NA')

dataset['State'] = State

dataset['State'].value_counts()

fig = plt.figure(figsize=(10, 7))
NA = (dataset.loc[dataset['State'] == "NA"])['State'].count()
Assign = ((dataset.loc[dataset['State'] != "NA"])['State'].count())
labels = ['Successful','Unsuccessful']
plt.xlabel('DETECTION', fontsize = 15) 
plt.ylabel('Total Vehicles', fontsize = 15) 
values = [Assign,NA]
print(values)
plt.title('Indian License Plate Detection')
plt.bar(labels,values)
plt.show()

df = dataset[dataset['State'] != "NA"]
fig = plt.figure(figsize=(32, 12))
ax=sns.countplot(x='State',data=df);   
plt.title('DETECTED INDIAN LICENSE PLATES', fontsize = 20) 
plt.xlabel('States', fontsize = 20) 
plt.ylabel('Total Vehicles', fontsize = 20) 
plt.show()
