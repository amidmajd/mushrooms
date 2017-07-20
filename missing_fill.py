import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree


data = pd.read_csv('data_cleaned.csv')
target = data['class']
data = data.drop('class', axis=1)
# this delete cause coloration between
# predicted missing values and target doesn't occur


# 28
cols = [
       'cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-color',
       'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
       'stalk-color-above-ring', 'stalk-color-below-ring',
       'veil-color', 'ring-type', 'spore-print-color', 'population', 'habitat',
       'b0', 'b1', 'ga0', 'ga1', 'gs0', 'gs1', 'gsz0', 'gsz1', 'ss0', 'ss1',
       'rn0', 'rn1', 'rn2'
]




# all FT labeled but stalk-root did not
# then after classifying nan and not-nan we are going to label it
data['stalk-root'] = data['stalk-root'].replace({'c': 1, 'e': 2, 'r': 0, 'b': 3})
# tmp = data.groupby('stalk-root').size()
# tmp = [(x,tmp[x]) for x in tmp.index]
# tmp = sorted(tmp, key=lambda x: x[1])
#
# tmp = { tmp[i][0]:i
#         for i in range(len(tmp))
# }
# print(tmp)

miss = data['stalk-root']

nan = data[pd.isnull(miss)]  # nan_len = 2480
nan = nan.drop('stalk-root', axis=1)

not_nan = data[~pd.isnull(miss)]



y_train = not_nan['stalk-root']
x_train = not_nan.drop('stalk-root', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=.2)

model = DecisionTreeClassifier(max_depth=27)
model.fit(x_train, y_train)
print('model score :', model.score(x_test, y_test))

nan_prediction = model.predict(nan)

data['stalk-root'][pd.isnull(data['stalk-root'])] = nan_prediction

# print(data.isnull().sum())

tmp_data = pd.concat([target, data], axis=1)
tmp_data.to_csv('data_final.csv', index=False)


file = open('miss_fill_tree.dot', 'w')
tree.export_graphviz(model, out_file=file)
file.close()
