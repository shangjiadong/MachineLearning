import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import time
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import linear_model, datasets
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

train = pd.read_csv('train.csv', na_values=-1)
test = pd.read_csv('test.csv', na_values=-1)

print(train.shape)
print(test.shape)

import seaborn as sns
cor = train.corr()

# There is no correlation between Calc and other features, we decide to get rid of them
drop_column = list(train.columns[train.columns.str.startswith('ps_calc_')])
train = train.drop(drop_column, axis=1)  
test = test.drop(drop_column, axis=1)

def missing_features(df):
    missings = pd.DataFrame([], columns=['feature', 'num_recoreds', 'percentage'])
    total_rows = df.shape[0]
    index = 0
    for feature in list(df):
        total_nulls = df[feature].isnull().sum()
        if total_nulls > 0:
            percentage = total_nulls / total_rows
            missings.loc[index] = [feature, total_nulls, percentage]
            index += 1
    missings = missings.sort_values('num_recoreds', ascending=False)
    return missings

missings = missing_features(train)
'''missings.plot(x='feature', y='num_recoreds', kind='bar', )
plt.show()'''

for i, feature in enumerate(list(train.drop(['id'], axis=1))):
    if train[feature].isnull().sum() > 0:
        train[feature].fillna(train[feature].mode()[0],inplace=True)

for i, feature in enumerate(list(test.drop(['id'], axis=1))):
    if test[feature].isnull().sum() > 0:
        test[feature].fillna(test[feature].mode()[0],inplace=True)

testID = test['id'].values
train_target= train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

def binarize(train):
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
    ps_car_11_b = pd.get_dummies(train['ps_car_11_cat'])
    ps_car_11 = preprocessing.scale(train['ps_car_11'])
    ps_car_12 = preprocessing.scale(train['ps_car_12'])
    ps_car_13 = preprocessing.scale(train['ps_car_13'])
    ps_car_14 = preprocessing.scale(train['ps_car_14'])
    ps_car_15 = preprocessing.scale(train['ps_car_15'])

    train_bin = np.column_stack((ps_ind_01, ps_ind_02, ps_ind_03,
    ps_ind_04,ps_ind_05,ps_ind_06,ps_ind_07,ps_ind_08,ps_ind_09,ps_ind_10,ps_ind_11,
    ps_ind_12,ps_ind_13,ps_ind_14,ps_ind_15,ps_ind_16,
    ps_ind_17,ps_ind_18,ps_reg_01,ps_reg_02,ps_reg_03,
    ps_car_01,ps_car_02,ps_car_03,ps_car_04,ps_car_05,
    ps_car_06,ps_car_07,ps_car_08,ps_car_09,ps_car_10,
    ps_car_11_b,ps_car_11,ps_car_12,ps_car_13,ps_car_14,
    ps_car_15))

    return train_bin

train = binarize(train)
test = binarize(test)

print(train.shape, test.shape)

# from https://www.kaggle.com/mashavasilenko/
# porto-seguro-xgb-modeling-and-parameters-tuning
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    #print all
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    #print giniSum
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

x_train, x_dev, y_train, y_dev = train_test_split(train, train_target, test_size = 0.1, random_state = 42)
print(x_train.shape, x_dev.shape)

# Select features using xgboost:
model  = XGBClassifier()
model.fit(x_train, y_train)
selection = SelectFromModel(model, prefit=True)
x_train = selection.transform(x_train)
x_dev = selection.transform(x_dev)
test = selection.transform(test)
print x_train.shape

poly_2 = PolynomialFeatures(2)
x_train = poly_2.fit_transform(x_train)
x_dev = poly_2.fit_transform(x_dev)
test = poly_2.fit_transform(test)
print x_train.shape
'''model2 = XGBClassifier()
model2.fit(x_train, y_train)'''

print "Training..."
seed = 7
np.random.seed(seed)

'''
model = Sequential()
model.add(Dense(528, input_dim = 528, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(np.array(x_train), np.array(y_train), nb_epoch = 5, batch_size = 64, verbose = 1)'''




# Logistic Regression
'''start_time = time.clock()
#log_Reg = BalancedBaggingClassifier(base_estimator= linear_model.LogisticRegression(C = 1e10, class_weight = {0:1, 1:1}), ratio={0:3, 1:1} ,replacement=True, random_state=0, max_samples=0.1, n_estimators = 10)
#log_Reg = AdaBoostClassifier(base_estimator=linear_model.LogisticRegression(C = 1e10), n_estimators=20)
log_Reg = linear_model.LogisticRegression(C = 1e10) # Default C = 1, but we can tune this hyper parameter
log_Reg.fit(x_train, y_train)
predict_dev_logi = log_Reg.predict_proba(x_dev)
print "Logistic regression, dev:"
print sum(predict_dev_logi >= 0.5) / y_dev.shape[0]
#print predict_dev_logi[:100]
predict_dev_logi_label = log_Reg.predict(x_dev)
print "dev error, logistic regression is", sum(predict_dev_logi_label == y_dev) / y_dev.shape[0]
print confusion_matrix(y_dev, predict_dev_logi_label)
predict_test_logi = log_Reg.predict_proba(test)
print "Logistic regression, test:"
print sum(predict_test_logi >= 0.5) / test.shape[0]
#print predict_test_logi[:100]
end_time = time.clock()
print "The time used on Logistic regression is", end_time - start_time'''


# Naive Bayes (probability calibration needs work)
# Use GaussianNB for now. After binarize features, should use BernoulliNB
'''start_time = time.clock()
gnb = GaussianNB()
#gnb = BalancedBaggingClassifier(base_estimator= GaussianNB(), ratio={0:1, 1:1}, replacement=True, random_state=0, max_samples=0.3, n_estimators= 3)
gnb.fit(x_train, y_train)
predict_gnb_dev = gnb.predict_proba(x_dev)
print "Naive Bayes, Gaussian feature, dev:"
print sum(predict_gnb_dev >= 0.5) / y_dev.shape[0]
print predict_gnb_dev[:100]
predict_dev_NB_label = gnb.predict(x_dev)
print sum(predict_dev_NB_label == y_dev)
print "dev errror of NB, Gaussian feature is", sum(predict_dev_NB_label == y_dev) / y_dev.shape[0]
print confusion_matrix(y_dev, predict_dev_NB_label)
gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method='isotonic')
gnb_isotonic.fit(x_train, y_train)
prob_dev_isotonic = gnb_isotonic.predict_proba(x_dev)
print prob_dev_isotonic[:100]
prob_test_isotonic = gnb_isotonic.predict_proba(test)
print prob_test_isotonic[:100]
print "Naive Bayes, Gaussian feature, test:"
predict_gnb_test = gnb.predict_proba(test)
print sum(predict_gnb_test >= 0.5) / test.shape[0]
print predict_gnb_test[:100]'''

from sklearn.neural_network import MLPClassifier
clf = sklearn.neural_network.MLPClassifier(solver='adam', alpha = 1e-5, hidden_layer_sizes=(60, 2), learning_rate = 'adaptive', random_state = 1, verbose = 1)
clf.fit(x_train, y_train)
clf.score(x_dev, y_dev)
results = clf.predict(test)

'''predict = model.predict(np.array(x_dev), batch_size = 64)
print("gini score on dev set:", gini_normalized(y_dev, predict))'''

'''predict = predict_dev_logi
print("gini score on dev set:", gini_normalized(y_dev, predict))

test = pd.read_csv('test.csv')
ID = test['id']
ID = np.array(ID)
with open('submission_noise_removed_lr.csv', 'w') as outfile:
    for idx, prob in zip(ID, predict_test_logi[:, 1]):
        outfile.write(str(idx) + ',' + str(prob) + '\n')
end_time = time.clock()'''

sub = pd.DataFrame()
sub['id'] = testID
sub['target'] = clf.predict_proba(np.array(test))
sub.to_csv('submission_mlp.csv', float_format='%.6f', index=False)

