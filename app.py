import numpy as np 
import pandas as pd 

df = pd.read_csv("Social_Network_Ads.csv")
# print(df.head(2)) 

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['Gender'] = lb.fit_transform(df['Gender'])

print(df.head(2)) 

x = df.drop(columns = ['Purchased'])
y = df['Purchased']

from sklearn.model_selection import train_test_split 

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42) 

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train , y_train) 

y_pred = lr.predict(x_test) 

from sklearn.metrics import accuracy_score 

print(accuracy_score(y_test , y_pred))