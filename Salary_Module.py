#import numpy to perform maths
import numpy as np
#this code is to import pandas
import pandas as pd
#this code is to import matplotlib library
import matplotlib.pyplot as plt

#creating a variable to store the data
salary_data=pd.read_csv('Salary_Data.csv')

#creating a variable x to store the dependent column values
x=salary_data.iloc[:,0:1].values

#creating a variable y to store the independent column values
y=salary_data.iloc[:,1:2].values

#splitting the variable
from sklearn.model_selection import train_test_split

#splitting the variable
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

# importing LinearRegression
from sklearn.linear_model import LinearRegression

#creating a variable to store LinearRegression
salary_module=LinearRegression()

#fitting x and y train
salary_module.fit(x_train,y_train)

#creating a variable to store prediction module
salary_prediction=salary_module.predict(x_test) 
salary_prediction


