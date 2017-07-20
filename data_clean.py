import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn
import os
seaborn.set(style='darkgrid', color_codes=True)


data = pd.read_csv('./mushrooms.csv')

# data.isnull().sum()
'''cols
       'stalk-root',

'''
# print(data.describe())


data['class'] = data['class'].replace({'e':0, 'p':1})   # target



# tmp = data.groupby('stalk-root').size()
# tmp = [(x,tmp[x]) for x in tmp.index]
# tmp = sorted(tmp, key=lambda x: x[1])
# tmp = { tmp[i][0]:i
#         for i in range(len(tmp))
# }
# print(tmp)


data['cap-shape'] = data['cap-shape'].replace({'x':5, 'f':4, 'k':3, 'b':2, 's':1, 'c':0})
data['cap-surface'] = data['cap-surface'].replace({'y':3, 's':2, 'f':1, 'g':0})
data['cap-color'] = data['cap-color'].replace({'p': 3, 'e': 7, 'c': 2, 'u': 1, 'y': 6, 'n': 9, 'b': 4, 'r': 0, 'g': 8, 'w': 5})

data['b0'] = np.zeros(len(data)) # t
data['b1'] = np.zeros(len(data)) # f
data['b0'][data['bruises'] == 't'] = 1.0
data['b1'][data['bruises'] == 'f'] = 1.0
# [('t', 3376), ('f', 4748)]
data = data.drop('bruises', axis=1)

data['odor'] = data['odor'].replace({'c': 1, 'f': 7, 's': 5, 'n': 8, 'p': 2, 'm': 0, 'y': 6, 'l': 4, 'a': 3})

data['ga0'] = np.zeros(len(data))
data['ga1'] = np.zeros(len(data))
data['ga0'][data['gill-attachment']=='a'] = 1.0
data['ga1'][data['gill-attachment']=='f'] = 1.0
# [('a', 210), ('f', 7914)]
data = data.drop('gill-attachment', axis=1)

data['gs0'] = np.zeros(len(data))
data['gs1'] = np.zeros(len(data))
data['gs0'][data['gill-spacing']=='w'] = 1.0
data['gs1'][data['gill-spacing']=='c'] = 1.0
# [('w', 1312), ('c', 6812)]
data = data.drop('gill-spacing', axis=1)

data['gsz0'] = np.zeros(len(data))
data['gsz1'] = np.zeros(len(data))
data['gsz0'][data['gill-size']=='n'] = 1.0
data['gsz1'][data['gill-size']=='b'] = 1.0
# [('n', 2512), ('b', 5612)]
data = data.drop('gill-size', axis=1)

data['gill-color'] = data['gill-color'].replace({'r': 0, 'k': 4, 'g': 7, 'u': 5, 'y': 2, 'e': 3,
                                                 'h': 6, 'p': 10, 'o': 1, 'b': 11, 'n': 8, 'w': 9})

data['ss0'] = np.zeros(len(data))
data['ss1'] = np.zeros(len(data))
data['ss0'][data['stalk-shape']=='e'] = 1.0
data['ss1'][data['stalk-shape']=='t'] = 1.0
# [('e', 3516), ('t', 4608)]
data = data.drop('stalk-shape', axis=1)

data['stalk-surface-above-ring'] = data['stalk-surface-above-ring'].replace({'k': 2, 's': 3, 'y': 0, 'f': 1})
data['stalk-surface-below-ring'] = data['stalk-surface-below-ring'].replace({'f': 1, 'y': 0, 'k': 2, 's': 3})
data['stalk-color-above-ring'] = data['stalk-color-above-ring'].replace({'p': 7, 'w': 8, 'e': 2, 'c': 1, 'b': 4,
                                                                         'y': 0, 'n': 5, 'o': 3, 'g': 6})

data['stalk-color-below-ring'] = data['stalk-color-below-ring'].replace({'g': 6, 'y': 0, 'e': 2, 'n': 5, 'o': 3,
                                                                        'c': 1, 'b': 4, 'w': 8, 'p': 7})

# 'veil-type' DELETED because of equality in all data
data = data.drop('veil-type', axis=1)

data['veil-color'] = data['veil-color'].replace({'o': 2, 'y': 0, 'n': 1, 'w': 3})

data['rn0'] = np.zeros(len(data))
data['rn1'] = np.zeros(len(data))
data['rn2'] = np.zeros(len(data))
data['rn0'][data['ring-number'] == 'o'] = 1.0
data['rn1'][data['ring-number'] == 't'] = 1.0
data['rn2'][data['ring-number'] == 'n'] = 1.0
# [('n', 36), ('t', 600), ('o', 7488)]
data = data.drop('ring-number', axis=1)

data['ring-type'] = data['ring-type'].replace({'f': 1, 'n': 0, 'p': 4, 'l': 2, 'e': 3})
data['spore-print-color'] = data['spore-print-color'].replace({'n': 7, 'h': 5, 'o': 1, 'u': 2, 'y': 3,
                                                               'r': 4, 'b': 0, 'k': 6, 'w': 8})

data['population'] = data['population'].replace({'a': 1, 'n': 2, 'c': 0, 's': 3, 'y': 4, 'v': 5})
data['habitat'] = data['habitat'].replace({'g': 5, 'd': 6, 'u': 2, 'w': 0, 'p': 4, 'm': 1, 'l': 3})


# missing data ==> ((((stalk-root))))
data['stalk-root'] = data['stalk-root'].replace({'?': np.nan})


data.to_csv('data_cleaned.csv', index=False)


# plt.scatter(data['cap-surface'][target==0].index,data['cap-surface'][target==0], c='b',s=100)
# plt.scatter(data['cap-surface'][target==1].index,data['cap-surface'][target==1] , c='r',s=100)


# plt.show()

