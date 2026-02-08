# import libraries
import pandas as pd
import numpy as np 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load data
data = sns.load_dataset("iris")

# labels and features

X= data.drop(columns="species")
y= data["species"]

# split data for traing and testing
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

# load model
model = KNeighborsClassifier()

# fit data into model
model.fit(X_train , y_train)

# check accuracy 
accuracy = model.score(X_test , y_test)
accuracy

# predict on new data
def predict_species (features) : 
    prediction = model.predict(pd.DataFrame([features] , columns=X.columns))
    return prediction[0]