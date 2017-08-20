import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras import *
from sklearn.model_selection import train_test_split
from termcolor import colored
sns.set()


data = pd.read_csv('data_final.csv')
target = data['class']
data = data.drop('class', axis=1)
c0 = np.zeros((len(data),1))
c1 = np.zeros((len(data),1))
c0[target==0] = 1
c1[target==1] = 1
C = np.append(c0,c1,axis=1)
target = pd.DataFrame(C,columns=['c0','c1'])
del c0, c1, C


cols = ['cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-color',
       'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color',
       'ring-type', 'spore-print-color', 'population', 'habitat', 'b0', 'b1',
       'ga0', 'ga1', 'gs0', 'gs1', 'gsz0', 'gsz1', 'ss0', 'ss1', 'rn0', 'rn1',
       'rn2']


model = models.Sequential()
model.add(layers.Dense(units=40, input_dim=len(data.columns), activation='sigmoid'))
model.add(layers.Dense(units=50, activation='relu'))
model.add(layers.Dense(units=50, activation='relu'))
model.add(layers.Dense(units=2, activation='sigmoid'))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.2)

model.fit(x_train, y_train,validation_split=0.1 ,epochs=200, batch_size=50,
          callbacks=[callbacks.TensorBoard(log_dir='./logs', batch_size=50,
                                            write_graph=True, write_grads=True)])

print(colored(model.evaluate(x_test, y_test, batch_size=len(x_test)), 'red', attrs=['bold']))
# model.save(filepath='NN_model.h5')
