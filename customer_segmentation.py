#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 12:04:28 2021

@author: faizeerturk

"""

import numpy as np
import pandas as pd

df = pd.read_csv('marketing_campaign.csv', sep='\t')

#PREPROSSESING and CLEANING
print("Number of datapoints:", len(df))

df.info()     #gives the information with 28 column and data type 
df.describe() #it gives total count of columnn ,mean, std, min, max ,vs

df=df.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(df))

#enrollinng dates 
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
dates = []
for i in df["Dt_Customer"]:
    i = i.date()
    dates.append(i)  
print("The newest customer's enrolment date :",max(dates))
print("The oldest customer's enrolment date :",min(dates))

print("Total categories in the feature Marital_Status:\n", df["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", df["Education"].value_counts())

#More meaningful columns adding
df["Age"] = 2021-df["Year_Birth"]
df["TotalSpent"] = df["MntWines"]+ df["MntFruits"]+df["MntMeatProducts"]+ df["MntFishProducts"]+ df["MntSweetProducts"]+ df["MntGoldProds"]
df["Children"]=df["Kidhome"]+df["Teenhome"]
df["Living"]=df.Marital_Status.replace({'Together': 'Partner','Single':'Alone',
                                                           'Married': 'Partner',
                                                           'Divorced': 'Alone',
                                                           'Widow': 'Alone', 
                                                           'Alone': 'Alone',
                                                           'Absurd': 'Alone',
                                                          'YOLO': 'Alone'})

df["TotalPerson"]=df.Marital_Status.replace({'Together': 2,'Single':1,
                                                           'Married': 2,
                                                           'Divorced': 1,
                                                           'Widow': 1, 
                                                           'Alone': 1,
                                                           'Absurd': 1,
                                                          'YOLO': 1})
print("Total categories in the feature Marital_Status:\n", df["Living"].value_counts(), "\n")

df["Family_Size"] = df["TotalPerson"]+ df["Children"]
df["Is_Parent"] = np.where(df.Children> 0, 1, 0) #last parameter is either both or neither of x and y
#This is changing education column from scract
df["Education"]=df["Education"].replace({"Basic":0,"2n Cycle":0, "Graduation":1, "Master":2, "PhD":2})
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
dropCol = [ "Marital_Status","Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
df = df.drop(dropCol, axis=1)
df.describe()

#There is 3 person whose age is grather than 120 so I thnik its not true I will drop them
print("The oldest customer :",max(df.Age)) # 128
df = df[(df["Age"]<120)]
print("The oldest customer :",max(df.Age)) #now 81 is the oldest which is much proper

#There is an noisy data in income column which is much more higher than other customers income so I will drop it too.
print("The outlier max ıncome  :",max(df.Income))
df = df[(df["Income"]<170000)]
print("The proper max ıncome  :",max(df.Income))

print("The total number of data-points after removing the outliers are:", len(df))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df ['Living'] = le.fit_transform(df[['Living']])

#from sklearn.preprocessing import MinMaxScaler
#mms=MinMaxScaler()
#df=pd.DataFrame(mms.fit_transform(df),columns=df.columns)

from sklearn.cluster import KMeans
km=KMeans(n_clusters=4)
km.fit(df)
segments=km.predict(df)
df['segments_kmean']=segments

#for elbos technique finding best numbers of cluster
#also for showing the score how many cluster provide best result
from sklearn.metrics import silhouette_score

print("-----K-means Clustering----")
l=[]
for i in range(2,6):
    km=KMeans(n_clusters=i)
    km.fit(df)
    segments=km.predict(df)
    ss = silhouette_score(df,segments)
    l+=[ss]
    print("score:",i, " many clusters ",ss)

distance=pd.DataFrame(data=l)
distance.plot()



from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=4)
segments2=hc.fit_predict(df)
df['segments_hc']= segments2

print("-----AgglomerativeClustering----")
l=[]
for i in range(2,6):
    hc=AgglomerativeClustering(n_clusters=i)
    segments=hc.fit_predict(df)
    ss = silhouette_score(df,segments)
    l+=[ss]
    print("score:",i, " many clusters ",ss)
    
import matplotlib.pyplot as plt #matplotlib library for plotting
import seaborn as sns           #seaborn library for plotting


Person = [ "Age", "Children", "Is_Parent", "Education","Family_Size","Recency"]

for i in Person:
    plt.figure()
    sns.jointplot(x=df[i], y=df["TotalSpent"], hue =df['segments_kmean'])
    sns.jointplot(x=df[i], y=df["TotalSpent"], hue =df['segments_kmean'],kind="scatter")
    sns.jointplot(x=df[i], y=df["TotalSpent"], hue =df['segments_kmean'],kind="kde")
    sns.jointplot(x=df[i], y=df["TotalSpent"], hue =df['segments_kmean'],kind="hist")
    sns.jointplot(x=df[i], y=df["Income"], hue =df['segments_kmean'],kind="kde")
    sns.jointplot(x=df[i], y=df["Income"], hue =df['segments_kmean'],kind="hist")
    plt.show()

for i in Person:
    plt.figure(figsize=(10,5))
    sns.swarmplot(x=i, y="TotalSpent", hue='segments_kmean' ,data=df)
    plt.show()

sns.countplot(df['Education'])
plt.show()

sns.pairplot(df,kind='reg')
plt.show()

corr = df.corr()
plt.figure(figsize=(40,40))
sns.heatmap(corr, annot=True, cmap='Greens');

plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)
plt.show()

lets_plot = ['Income', 'TotalSpent' , 'Age','Education','Family_Size','Is_Parent']
sns.pairplot(df[lets_plot], hue = 'Is_Parent')

lets_plot = ['Income', 'TotalSpent' ,'Age', 'Family_Size','Is_Parent','Education']
sns.pairplot(df[lets_plot], hue = 'Education')