from __future__ import division
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import linear_model, datasets
from sklearn.naive_bayes import GaussianNB
#=========================================================================================
# The following script is for data preprossing

train = pd.read_csv('train.csv')
# create the train file
'''x_train_all = train.drop('id', axis = 1)
x_train_all = x_train_all.drop('target', axis = 1)
y_train_all = train['target']
# split the data into train and dev set with percentage 0.1
# now we have train and dev set
x_train, x_dev, y_train, y_dev = train_test_split(x_train_all, y_train_all, test_size=0.10, random_state=42)
print x_train.shape, x_dev.shape
# check how many positive target
y_train_all.describe(include = ['target'])

# count    595212.000000
# mean          0.036448
# std           0.187401
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           0.000000
# max           1.000000
# Name: target, dtype: float64

# 3.6448% are 1s, the rest are 0s
# there are 595212 examples, we draw 10% as the toy set
# then I would need 59521 examples, include 2170 positive examples, and 57352 negative examples
# the following code generates the toy set
positive_example = train[train['target'] == 1]
negative_example = train[train['target'] == 0]
toy_positive = positive_example.ix[random.sample(positive_example.index, 2170)]
toy_negative = negative_example.ix[random.sample(negative_example.index, 57352)]
toy = pd.concat([toy_positive, toy_negative])
x_toy = toy.drop('id', axis = 1)
x_toy = x_toy.drop('target', axis = 1)
y_toy = toy['target']
x_train_toy, x_dev_toy, y_train_toy, y_dev_toy = train_test_split(x_toy, y_toy, test_size = 0.10, random_state = 42)
print x_train_toy.shape, y_train_toy.shape, x_dev_toy.shape, y_dev_toy.shape

# Following code writes the data to file
# x_train_toy.to_csv('x_train_toy.txt', header=True, index=None, sep=' ', mode='a')
# y_train_toy.to_csv('y_train_toy.txt', header=True, index=None, sep=' ', mode='a')
# x_dev_toy.to_csv('x_dev_toy.txt', header=True, index=None, sep=' ', mode='a')
# y_dev_toy.to_csv('y_dev_toy.txt', header=True, index=None, sep=' ', mode='a')
# x_train.to_csv('x_train.txt', header=True, index=None, sep=' ', mode='a')
# y_train.to_csv('y_train.txt', header=True, index=None, sep=' ', mode='a')
# x_dev.to_csv('x_dev.txt', header=True, index=None, sep=' ', mode='a')
# y_dev.to_csv('y_dev.txt', header=True, index=None, sep=' ', mode='a')
'''
#=========================================================================================
# Data read-in
x_train_toy = pd.read_csv('x_train_toy.txt', sep = ' ')
y_train_toy = pd.read_csv('y_train_toy.txt', sep = ' ')
x_dev_toy = pd.read_csv('x_dev_toy.txt', sep = ' ')
y_dev_toy = pd.read_csv('y_dev_toy.txt', sep = ' ')

x_train = pd.read_csv('x_train.txt', sep = ' ')
y_train = pd.read_csv('y_train.txt', sep = ' ')
x_dev = pd.read_csv('x_dev.txt', sep = ' ')
y_dev = pd.read_csv('y_dev.txt', sep = ' ')
#=========================================================================================
# Convert to np numberical arrays
x_train = np.array(x_train).astype(float)
y_train = np.array(y_train).astype(np.int32).flatten()
x_dev = np.array(x_dev).astype(float)
y_dev = np.array(y_dev).astype(np.int32).flatten()

x_train_toy = np.array(x_train_toy).astype(float)
y_train_toy = np.array(y_train_toy).astype(np.int32).flatten()
x_dev_toy = np.array(x_dev_toy).astype(float)
y_dev_toy = np.array(y_dev_toy).astype(np.int32).flatten()

test = pd.read_csv('test.csv')
ID = test['id']
test = test.drop('id', axis = 1)

test = np.array(test).astype(float)

seed = 7
np.random.seed(seed)

#=====================================================================
# Feature engineering: Should binarize categorical features, standardize/bin/binarize numerical features, try quadratic/cube models, etc.
#=====================================================================
# 3 Methods

# Deep learning:
'''model = Sequential()
model.add(Dense(57, input_dim = 57, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, nb_epoch = 5, batch_size = 64, verbose = 1)

predict = model.predict(x_dev, batch_size = 64)
proba = model.predict_proba(x_dev, batch_size = 64)

predict_test = model.predict(test, batch_size = 64)

print predict_test
ID = np.array(ID).reshape(892816, 1)
submission = np.stack((ID, predict_test), axis = -1)
with open('submission.csv', 'w') as outfile:
    for idx, prob in zip(ID, predict_test):
        outfile.write(str(idx) + ',' + str(prob) + '\n')'''

# test = test.drop('id', axis = 1)
# test = np.array(test).astype(float)

seed = 7
np.random.seed(seed)
#==========================================================================================================
# Logistic Regression
start_time = time.clock()
log_Reg = linear_model.LogisticRegression(C = 1e10) # Default C = 1, but we can tune this hyper parameter
#log_Reg.fit(x_train, y_train)
log_Reg.fit(x_train, y_train)
predict_dev_logi = log_Reg.predict_proba(x_dev)
print "Logistic regression, dev:"
print sum(predict_dev_logi >= 0.5) / y_dev.shape[0]
#print predict_dev_logi[:100]
predict_dev_logi_label = log_Reg.predict(x_dev)
print "dev error, logistic regression is", sum(predict_dev_logi_label == y_dev) / y_dev.shape[0]
predict_test_logi = log_Reg.predict_proba(test)
print "Logistic regression, test:"
print sum(predict_test_logi >= 0.5) / test.shape[0]
#print predict_test_logi[:100]
end_time = time.clock()
print "The time used on Logistic regression is", end_time - start_time
#==========================================================================================================
# Naive Bayes (probability calibration needs work)
# Use GaussianNB for now. After binarize features, should use BernoulliNB
start_time = time.clock()
gnb = GaussianNB()
gnb.fit(x_train, y_train)
predict_gnb_dev = gnb.predict_proba(x_dev)
print "Naive Bayes, Gaussian feature, dev:"
print sum(predict_gnb_dev >= 0.5) / y_dev.shape[0]
#print predict_gnb_dev[:100]
predict_dev_NB_label = gnb.predict(x_dev)
print sum(predict_dev_NB_label == y_dev)
print "dev errror of NB, Gaussian feature is", sum(predict_dev_NB_label == y_dev) / y_dev.shape[0]
print "Naive Bayes, Gaussian feature, test:"
predict_gnb_test = gnb.predict_proba(test)
print sum(predict_gnb_test >= 0.5) / test.shape[0]
#print predict_gnb_test[:100]
end_time = time.clock()
print "The time used on Naive Bayes, Gaussian feature is", end_time - start_time


