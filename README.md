# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and remove unnecessary attributes such as serial number and salary.

2. Convert all categorical attributes into numerical values and separate the independent variables X and dependent variable Y.

3. Initialize the parameter vector θ with random values.

4. Compute the hypothesis using the sigmoid function:
<img width="319" height="89" alt="image" src="https://github.com/user-attachments/assets/1c1f526c-4e7a-4c71-bcf7-2688131a6bc0" />

5. Define the logistic loss function and update the parameters using gradient descent:
<img width="376" height="88" alt="image" src="https://github.com/user-attachments/assets/5b6572be-f862-4fbf-b157-70dba547c596" />

 Repeat this step for a fixed number of iterations.


6. Use the optimized parameters θ to predict the output class by applying a threshold (0.5) and evaluate the model accuracy.

## Program:
```PYTHON
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SRI SRINIVASAN K
RegisterNumber:  212224220104
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the columns
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

#selecting the features and labels
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
#display independent variable
Y

#initialize the model parameters
theta=np.random.randn(X.shape[1])
y=Y
#define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))
#define the loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

#defining the gradient descent algorithm.
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
#train the model
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
#makeprev \dictions
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)


accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```

## Output:

### Read the file and display

<img width="1204" height="449" alt="image" src="https://github.com/user-attachments/assets/0ebea9c5-d4c2-40d3-8e88-85f7d10ca11b" />

### Categorizing columns

<img width="331" height="310" alt="image" src="https://github.com/user-attachments/assets/1738b586-11d0-437c-adbf-3a480fe99b50" />

### Labelling columns and displaying dataset

<img width="982" height="436" alt="image" src="https://github.com/user-attachments/assets/0ac51ad7-96ca-453c-8640-f23fe48a5530" />

### Display dependent variable

<img width="729" height="229" alt="image" src="https://github.com/user-attachments/assets/3214111d-fd6b-4be5-b353-9c66dd432da0" />

### Printing accuracy

<img width="357" height="27" alt="image" src="https://github.com/user-attachments/assets/276d1197-1ca4-4396-b507-71c7a29ef1e7" />

### Printing Y

<img width="744" height="133" alt="image" src="https://github.com/user-attachments/assets/a2d41eea-a1e5-4ebf-a770-526af18bacfd" />

### Printing y_prednew

<img width="98" height="44" alt="image" src="https://github.com/user-attachments/assets/e8d20997-961e-4cb0-a84e-fd3fb9e07b02" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

