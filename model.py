from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import tree
import seaborn
seaborn.set()


data = pd.read_csv('data_final.csv')
target = data['class']
data = data.drop('class', axis=1)
cols = [
       'cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-color',
       'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color',
       'ring-type', 'spore-print-color', 'population', 'habitat', 'b0', 'b1',
       'ga0', 'ga1', 'gs0', 'gs1', 'gsz0', 'gsz1', 'ss0', 'ss1', 'rn0', 'rn1',
       'rn2'
]   # 28 cols


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

model = RandomForestClassifier(criterion='entropy', max_depth=28, n_estimators=100)
model.fit(x_train, y_train)

print('score :', model.score(x_test, y_test))
print('mean squared Error :', mean_squared_error(y_test, model.predict(x_test)))



feature_imp = [(cols[i], imp) for i,imp in enumerate(model.feature_importances_)]
feature_imp = sorted(feature_imp, key=lambda x: x[1])

fig, ax = plt.subplots()
# plt.hist([y for x, y in feature_imp], color='r', orientation='horizontal', bins=30)
# plt.yticks([x for x,y in feature_imp])
ax.barh(range(28),[y for x,y in feature_imp], color='r')
ax.set_yticks(range(28))
ax.set_yticklabels([x for x,y in feature_imp], fontsize=12)
fig.suptitle('Feature importance', fontsize=15)
fig.set_size_inches(16,8)
# plt.show()
plt.savefig('feature_importance.svg')


# i_tree = 0
# for tree_in_forest in model.estimators_:
#     with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
#         my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
#     i_tree = i_tree + 1