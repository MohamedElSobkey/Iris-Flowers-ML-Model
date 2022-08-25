# importing some important basic libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import iris dataset
df=pd.read_csv(r'iris.data')
#df=pd.read_csv(r'Index')



#view the data frame

df

df.info()

#checking for null values
df.isnull().sum()

df.columns

#Drop unwanted columns
df=df.drop(columns="5.1")

df


#Visualizations
#View the count plot of Iris-setosa feature using seaborn.
df['Iris-setosa'].value_counts()

sns.countplot(df['Iris-setosa']);


#Define x and y. x contains all the input variables such as independent features, and y should contain the dependent variable which is dependent on independent variables, the output.
x=df.iloc[:,:4]
y=df.iloc[:,4]

#Split the Data Into Train and Test Datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

#View their shapes
x_train.shape
x_test.shape
y_train.shape
y_test.shape

#Create the Model (Classification)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

#In the fit method, pass training datasets in it. x_train and y_train are the training datasets.
model.fit(x_train,y_train)

#Now predict the results using predict method.
y_pred=model.predict(x_test)

#View the results now
y_pred

# the accuracy of the model and view the confusion matrix.

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)

accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))
