# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the dataset from the CSV file and divide it into the independent feature (Hours) and the dependent target (Scores).
2. Partition the data into training and testing subsets, then build a Linear Regression model using the training portion.
3. Generate predictions for the test set using the trained model and compare them against the real scores.
4. Visualize the regression line along with training and test points, and calculate evaluation metrics such as MAE, MSE, and RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("student_scores.csv")

X = df[['Hours']]
Y = df['Scores']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print(Y_pred)

plt.scatter(X_test, Y_test, color='blue', label="Test Data")
plt.plot(X_test, Y_pred, color='green', label="Best Fit")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.legend()
plt.show()


mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error :", mae)
print("Mean Squared Error :", mse)
print("Root Mean Squared Error :", rmse)

Developed by: Joshua Abraham Phlip A
RegisterNumber: 25013744
*/
```

## Output:
<img width="632" height="502" alt="Screenshot 2025-12-04 at 9 48 49 PM" src="https://github.com/user-attachments/assets/946a34fe-7396-4555-bce8-b1008845ca62" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
