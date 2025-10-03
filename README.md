# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Set initial values for slope (m) and intercept (b) to zero, along with learning rate and number of iterations.

2.Compute the Mean Squared Error (MSE) between actual and predicted values using the current parameters.

3.Calculate partial derivatives of the cost function with respect to slope and intercept to determine the direction of descent.

4.Adjust the slope and intercept using gradient descent formula.

## Program
```
/*
Program to implement the linear regression using gradient descent.
Developed by: R.Sairam
RegisterNumber:  25000694
*/
```
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

df=pd.read_csv("50_Startups.csv")

X = df[['R&D Spend','Administration','Marketing Spend','State']]
y = df['Profit']

m, b = 0, 0

lr = 0.01

iteration = 1000

n = len(X)

lines = {}

for i in range(iteration):

y_pred = m * X + b    

error = y - y_pred
    
dm = (-2/n)*np.sum(X * error)
    
db = (-2/n)*np.sum(error)
    
m -= lr * dm

b -= lr * db
    
if i in [0,100, 200, 300, 400,500,600,700,800,900]:                

 lines[i] = (m, b)
        
if i % 100 == 0:  
    
 cost = np.mean(error**2)
    
print(f"Iteration {i}: m={m:.4f}, b={b:.4f}, cost={cost:.6f}")

final_predictions = np.dot(X, m) + b

final_cost = np.mean((y - final_predictions)**2)

print(f"Final cost (MSE): {final_cost:.6f}")

plt.scatter(X[:, 0], y, color="red", label="Data points", alpha=0.7, s=50)

plt.title("Gradient Descent Progress for Startup Profit Prediction")

plt.xlabel("R&D Spend")

plt.ylabel("Profit")

plt.legend()

plt.show()



## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
