import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

train = pd.read_csv('train.csv')
def binarize(train):
	column = [item for item in train]
	if 'target' in column:
		target = np.array(train['target'])
	else: 
		target = []
	ps_ind_01 = preprocessing.scale(train['ps_ind_01'])
	ps_ind_02 = pd.get_dummies(train['ps_ind_02_cat'])
	ps_ind_03 = preprocessing.scale(train['ps_ind_03'])
	ps_ind_04 = pd.get_dummies(train['ps_ind_04_cat'])
	ps_ind_05 = pd.get_dummies(train['ps_ind_05_cat'])
	ps_ind_06 = train['ps_ind_06_bin']
	ps_ind_07 = train['ps_ind_07_bin']
	ps_ind_08 = train['ps_ind_08_bin']
	ps_ind_09 = train['ps_ind_09_bin']
	ps_ind_10 = train['ps_ind_10_bin']
	ps_ind_11 = train['ps_ind_11_bin']
	ps_ind_12 = train['ps_ind_12_bin']
	ps_ind_13 = train['ps_ind_13_bin']
	ps_ind_14 = preprocessing.scale(train['ps_ind_14'])
	ps_ind_15 = preprocessing.scale(train['ps_ind_15'])
	ps_ind_16 = train['ps_ind_16_bin']
	ps_ind_17 = train['ps_ind_17_bin']
	ps_ind_18 = train['ps_ind_18_bin']
	ps_reg_01 = preprocessing.scale(train['ps_reg_01'])
	ps_reg_02 = preprocessing.scale(train['ps_reg_02'])
	ps_reg_03 = preprocessing.scale(train['ps_reg_03'])
	ps_car_01 = pd.get_dummies(train['ps_car_01_cat'])
	ps_car_02 = pd.get_dummies(train['ps_car_02_cat'])
	ps_car_03 = pd.get_dummies(train['ps_car_03_cat'])
	ps_car_04 = pd.get_dummies(train['ps_car_04_cat'])
	ps_car_05 = pd.get_dummies(train['ps_car_05_cat'])
	ps_car_06 = pd.get_dummies(train['ps_car_06_cat'])
	ps_car_07 = pd.get_dummies(train['ps_car_07_cat'])
	ps_car_08 = pd.get_dummies(train['ps_car_08_cat'])
	ps_car_09 = pd.get_dummies(train['ps_car_09_cat'])
	ps_car_10 = pd.get_dummies(train['ps_car_10_cat'])
	ps_car_11 = preprocessing.scale(train['ps_car_11'])
	ps_car_12 = preprocessing.scale(train['ps_car_12'])
	ps_car_13 = preprocessing.scale(train['ps_car_13'])
	ps_car_14 = preprocessing.scale(train['ps_car_14'])
	ps_car_15 = preprocessing.scale(train['ps_car_15'])
	ps_calc_01 = preprocessing.scale(train['ps_calc_01'])
	ps_calc_02 = preprocessing.scale(train['ps_calc_02'])
	ps_calc_03 = preprocessing.scale(train['ps_calc_03'])
	ps_calc_04 = preprocessing.scale(train['ps_calc_04'])
	ps_calc_05 = preprocessing.scale(train['ps_calc_05'])
	ps_calc_06 = preprocessing.scale(train['ps_calc_06'])
	ps_calc_07 = preprocessing.scale(train['ps_calc_07'])
	ps_calc_08 = preprocessing.scale(train['ps_calc_08'])
	ps_calc_09 = preprocessing.scale(train['ps_calc_09'])
	ps_calc_10 = preprocessing.scale(train['ps_calc_10'])
	ps_calc_11 = preprocessing.scale(train['ps_calc_11'])
	ps_calc_12 = preprocessing.scale(train['ps_calc_12'])
	ps_calc_13 = preprocessing.scale(train['ps_calc_13'])
	ps_calc_14 = preprocessing.scale(train['ps_calc_14'])
	ps_calc_15 = train['ps_calc_15_bin']
	ps_calc_16 = train['ps_calc_16_bin']
	ps_calc_17 = train['ps_calc_17_bin']
	ps_calc_18 = train['ps_calc_18_bin']
	ps_calc_19 = train['ps_calc_19_bin']
	ps_calc_20 = train['ps_calc_20_bin']
	train_process = np.column_stack((ps_ind_01, ps_ind_02, ps_ind_03, 
	ps_ind_04,ps_ind_05,ps_ind_06,ps_ind_07,ps_ind_08,ps_ind_09,ps_ind_10,ps_ind_11,
	ps_ind_12,ps_ind_13,ps_ind_14,ps_ind_15,ps_ind_16,
	ps_ind_17,ps_ind_18,ps_reg_01,ps_reg_02,ps_reg_03,
	ps_car_01,ps_car_02,ps_car_03,ps_car_04,ps_car_05,
	ps_car_06,ps_car_07,ps_car_08,ps_car_09,ps_car_10,
	ps_car_11,ps_car_11,ps_car_12,ps_car_13,ps_car_14,
	ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04,
	ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,
	ps_calc_10,ps_calc_11,ps_calc_12,ps_calc_13,ps_calc_14,
	ps_calc_15,ps_calc_16,ps_calc_17,ps_calc_18,ps_calc_19,ps_calc_20))
	return train_process, target

x_train_all_bin, y_train_all_bin = binarize(train)

x_train, x_dev, y_train, y_dev = train_test_split(x_train_all_bin, y_train_all_bin, test_size = 0.1, random_state = 42)
print(x_train.shape, x_dev.shape)

test = pd.read_csv('test.csv')
ID = test['id']
test, _ = binarize(test)

clf = sklearn.neural_network.MLPClassifier(solver='adam', alpha = 1e-5, hidden_layer_sizes=(60, 2), learning_rate = 'adaptive', random_state = 1, verbose = 1)
clf.fit(x_train, y_train)

clf.score(x_dev, y_dev)
pred_prob = clf.predict_proba(test)[:, 1]
pred_prob = pred_prob.reshape(892816, 1)
ID = np.array(ID).reshape(892816, 1)
submission = np.stack((ID, pred_prob), axis = -1)
with open('submission_MLCP.csv', 'w') as outfile:
    for idx, prob in zip(ID, pred_prob):
        outfile.write(str(idx) + ',' + str(prob) + '\n')