import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance

data = pd.read_csv('datasets/FIFA 2018 Statistics.csv')
y = (data['Man of the Match']=="Yes")

feature_names = [i for i in data.columns if data[i].dtype in [np.int64] ]
X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split( X, y, random_state=1)
model = RandomForestClassifier( random_state=0).fit(train_X, train_y)

perm = PermutationImportance( my_model, random_state)