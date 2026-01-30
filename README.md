## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Daffodil Irene.s
RegisterNumber: 212225100006 
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("ex3.csv")

x = data["R&D Spend"].values
y = data["Profit"].values


x = (x - np.mean(x)) / np.std(x)


w = 0.0          
b = 0.0          
alpha = 0.01    
epochs = 100
n = len(x)

losses = []


for i in range(epochs):
    
    y_hat = w * x + b

   
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    
    w = w - alpha * dw
    b = b - alpha * db


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")


plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()


print("Final Weight (w):", w)
print("Final Bias (b):", b)

```

## Output:

<img width="1240" height="579" alt="Screenshot 2026-01-30 143443" src="https://github.com/user-attachments/assets/256e1290-7096-42a2-be01-3dfd889622e5" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
