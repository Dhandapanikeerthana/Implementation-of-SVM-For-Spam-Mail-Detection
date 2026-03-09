# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.

2. Analyse the data.

3. Use modelselection and Countvectorizer to preditct the values.

4. Find the accuracy and display the result. 

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: Keerthana D
RegisterNumber: 212224040155 

```
```

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

## data

<img width="918" height="560" alt="image" src="https://github.com/user-attachments/assets/69927c6b-d13d-4762-8cd0-b26a6d9013b3" />

## confusion matrix

<img width="924" height="65" alt="image" src="https://github.com/user-attachments/assets/42a44036-710b-4c74-b044-ae3f1e868e8b" />

## accuracy
<img width="919" height="64" alt="image" src="https://github.com/user-attachments/assets/d4a1ca09-f5a8-4742-b14e-df52fec45ccf" />

## classification report

<img width="917" height="264" alt="image" src="https://github.com/user-attachments/assets/27fa53ec-9acf-4626-8b21-3263665f825c" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
