# -*- coding: utf-8 -*-
"""Copy of Breast Cancer Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q8tOQb8JXEJNS6IGvd-NlXmF2J1pTQuO

# **BREAST CANCER PATTERN DETECTION USING ML**

---

# Data Collection
"""

#Importing Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


#Loading the Dataset
dataset = pd.read_csv('data.csv')

#Checking the different keys of our dataset
dataset.keys()

dataset.shape

#Printing the first 5 rows of the dataframe for visualization
dataset.head(20)

"""# Data Preparation"""

dataset.info()

#Data Cleaning
data=dataset.drop(['id','diagnosis'], axis=1)
target=dataset['diagnosis']

data.shape

data.head(5)

target.replace(to_replace ="M", value ="1",inplace=True)

target.replace(to_replace ="B", value ="0",inplace=True)

"""# Data Exploration"""

#To identify count of Malignant and Benign Cancer
fig=plt.subplots(figsize=(10,7))
sns.countplot(target,data=dataset)

fig=plt.subplots(figsize=(10,7))
sns.kdeplot(data['symmetry_worst'])

fig=plt.subplots(figsize=(10,7))
sns.distplot(data['radius_mean'],kde=False,bins=30)

fig=plt.subplots(figsize=(10,7))
sns.distplot(data['texture_mean'],kde=True,bins=45,color='green')

#Checking for any missing value or null value
fig=plt.subplots(figsize=(10,7))
sns.heatmap(data.isnull(), cbar=False, yticklabels=False)

#Dividing the data into x and y for training the model
#X-Contains the features on which the model will classify the type of breast cancer
#y-Contains the original classifications
y=target
x = data[['texture_mean','area_mean','concavity_mean','area_se','concavity_se','fractal_dimension_se','smoothness_worst','concavity_worst', 'symmetry_worst','fractal_dimension_worst']]

from sklearn.model_selection import train_test_split

#Spliting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

"""# Choose a Model"""

# For Scaling values
from sklearn.preprocessing import StandardScaler

X_train.shape

X_test.shape

y_train.shape

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Intialising the ANN Model
model=Sequential()

#Adding the first hidden layer of our ANN Model
#units-Number of nodes we want to add in our hidden layer.
#kernel_intializer-The function is used to intialise the weights.
#input_dim=Number of nodes in the input layer
#activation=The Activation Function
#Dropout function is used to avoid overfitting 
#because overfitting fails in generalising the model
model.add(Dense(units=5,kernel_initializer='uniform',activation='relu',input_dim=10))
model.add(Dropout(0.2))

#Adding the Second hidden layer of our ANN Model
model.add(Dense(units=5,kernel_initializer='uniform',activation='relu'))
model.add(Dropout(0.2))

#Adding the output layer of our ANN Model
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#Compiling the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

print(X_train)

"""# Train the Model"""

y_train=y_train.apply(lambda x: float(x))

#Fitting the model to our dataset and specifying the number of batch_size and epochs
model.fit(X_train,y_train,epochs=100,batch_size=10)

"""# Evaluate the Model"""

#Making prediction of our model
prediction=model.predict(X_test)

#Evaluating the performance of our model
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

prediction=(prediction>0.5)

#Confusion Matrix
y_test_numpy = y_test.to_numpy().astype(np.int)
print(confusion_matrix(y_test_numpy,prediction))

#Classifiaction Report 
print(classification_report(y_test_numpy,prediction))

#Accuracy of our Model
print("Accuracy: "+ str((accuracy_score(y_test_numpy,prediction))*100))

model.save("model.h5")
import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb'))
