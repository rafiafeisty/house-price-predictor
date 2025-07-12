import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('Housing.csv')
df_encoded=pd.get_dummies(df,drop_first=True)

X=df_encoded.drop('price',axis=1)
Y=df_encoded['price']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=linear_model.LinearRegression()
model.fit(X_train,Y_train)
house_predict=model.predict(X_test)
error_rate=mean_squared_error(Y_test,house_predict)
print("error rate: ",error_rate)
print("weight: ",model.coef_)
print("intercept: ",model.intercept_)

sorted_index = Y_test.argsort()
Y_test_sorted = Y_test.iloc[sorted_index]
house_predict_sorted = house_predict[sorted_index]

plt.plot(Y_test_sorted.values, label="Actual")
plt.plot(house_predict_sorted, label="Predicted")
plt.xlabel("Sample Index")
plt.ylabel("House Price")
plt.legend()
plt.title("Actual vs Predicted (Line Plot)")
plt.show()
