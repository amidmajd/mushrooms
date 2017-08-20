import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


data = pd.read_csv('data_final.csv')
target = data['class']
data = data.drop('class', axis=1)

pca = PCA(n_components=3)
x = pca.fit_transform(data.values)

c0 = target[target==0].index
c1 = target[target==1].index

plt.scatter(x[c0,0], x[c0,1], marker='o', color='green',s=15)
plt.scatter(x[c1,0], x[c1,1], marker='x', color='red',s=15)
plt.show()
