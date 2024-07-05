import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import pickle

# Load the csv file
car = pd.read_csv("clean_cars.csv")

print(car.head())

X=car[['marque','modele','puissance_fiscale','kilometrage','annee','energie','boite']]
y=car['prix']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

ohe=OneHotEncoder()
ohe.fit(car[['marque','modele','energie','boite']])

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['marque','modele','energie','boite']),
                                    remainder='passthrough')
tree = DecisionTreeClassifier()

pipe_tree=make_pipeline(column_trans,tree)
pipe_tree.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(pipe_tree, open("model.pkl", "wb"))
