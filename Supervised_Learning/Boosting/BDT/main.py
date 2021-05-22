import os
import shutil
import logging
import argparse
import pandas as pd

dic = {}
dic['regression'] = [
    pd.DataFrame(
        data=[[1, 5, 20, 1.1], [2, 7, 30, 1.3],
              [3, 21, 70, 1.7], [4, 30, 60, 1.8]], 
        columns=['id', 'age', 'weight', 'label']
    ),
    pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])
]

dic['binary_cf'] = [
    pd.DataFrame(
        data=[[1, 5, 20, 0], [2, 7, 30, 0],
              [3, 21, 70, 1], [4, 30, 60, 1]],
        columns=['id', 'age', 'weight', 'label']
    ),
    pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])
]

dic['multi_cf'] = [
    pd.DataFrame(
        data=[[1, 5, 20, 0], [2, 7, 30, 0],
              [3, 21, 70, 1], [4, 30, 60, 1],
              [5, 30, 60, 2], [6, 30, 70, 2]],
        columns=['id', 'age', 'weight', 'label']
    ),
    pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])
]

dic['multi_cf'] = [
    pd.DataFrame(
        data=[[1, 5, 20, 0], [2, 7, 30, 0],
              [3, 21, 70, 1], [4, 30, 60, 1],
              [5, 30, 60, 2], [6, 30, 70, 2]], 
        columns=['id', 'age', 'weight', 'label']
    ),
    pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])
]

model = None
model_name = 'multi_cf' #'binary_cf' #'regression' #'binary_cf' 'multi_cf'

train_data = dic[model_name][0]
test_data = dic[model_name][1]



'''
if model_name == 'regression':
    model = GradientBoostingRegressor(
        learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
        min_samples_split=args.count, is_log=args.log, is_plot=args.plot
    )

if model_name == 'binary_cf':
    model = GradientBoostingBinaryClassifier(
        learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
        is_log=args.log, is_plot=args.plot
    )

if model_name == 'multi_cf':
    model = GradientBoostingMultiClassifier(
        learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth, 
        is_log=args.log,is_plot=args.plot
    )

model.fit(train_data)
model.predict(test_data)
'''