#Artificial Neural Network

#Installing Theno
#Installing tensorflow
#Installing Keras

#Part 1-Data Preprocessing
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:, 3:13].values
y=dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X =np.array(columnTransformer.fit_transform(X),dtype=np.float)

#Avoiding dummy variable trap
X=X[:,1:]

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Part 2-Let's make ANN!

#Importing Keras libraries and modules
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing ANN
classifier=Sequential()

#Adding the input layer and the  first  hidden  layer
classifier.add(Dense(6,activation='relu',input_dim=11))

#Adding the second hidden layer
classifier.add(Dense(6,activation='relu'))

#Adding the output layer
classifier.add(Dense(1,activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#Part 3-Making the predictions and evaluating the model

#Predicting the test set results
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)