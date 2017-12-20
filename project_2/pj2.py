from __future__ import division
import numpy as np
from sklearn import svm
from random import randrange
import scipy
import os
import itertools
import operator
import time
import timeit
from numpy import linalg
from sklearn.preprocessing import PolynomialFeatures
import copy
import matplotlib.pyplot as plt
os.chdir("C:/MachineLearning/pj2/pj2-data")

# read data
def readData(file, test = False, bias = True, num_idx = []):
	income_raw = np.array(np.genfromtxt(file, delimiter=",", dtype='S20'))
	#for i in range(num_idx.shape)
	num_age = income_raw[:, num_idx[0]].astype(np.int)
	num_hour = income_raw[:, num_idx[1]].astype(np.int)
	num_income = np.array(np.column_stack((num_age, num_hour)))
	if bias == True:
		bias = np.ones((income_raw.shape[0], 1))
		income_raw = np.concatenate((bias, income_raw), axis=1)
	if test == False:
		x = income_raw[:, :-1]
		y = income_raw[:, -1]
		return x, num_income, y
		#names="Age, WorkClass, Education, MaritalStatus, Occupation, Race, Sex, Hour, Country")
	else:
        #names = "Age, WorkClass, Education, MaritalStatus, Occupation, Race, Sex, Hour, Country, Target")
		return income_raw, num_income

# binarize features
def binarizeFeature(x, unique_list = []):
	binarized_list = []
	if unique_list == []:
		for i in range(x.shape[1]):
			unique_values = list(set(x[:, i]))
			bin_results = np.zeros(shape = (x.shape[0], len(unique_values)))
			for j, value in enumerate(x[:, i]):
				bin_results[j, unique_values.index(value)] = 1.0
			binarized_list.append(bin_results)
			unique_list.append(unique_values)
		x_bin = np.concatenate(binarized_list, axis=1)
		return x_bin, unique_list
	else:
		for i in range(len(unique_list)):
			unique_values = unique_list[i]
			bin_results = np.zeros(shape=(x.shape[0], len(unique_values)))
			for j, value in enumerate(x[:, i]):
				if value in unique_values:
					bin_results[j, unique_values.index(value)] = 1.0
			binarized_list.append(bin_results)
		x_bin = np.concatenate(binarized_list, axis = 1)
		return x_bin

# Generating labels w.r.t. y
def gen_label(y):
	y_bin = np.zeros((y.shape[0], 1))
	for i in range(y.shape[0]):
		if y[i] == ' >50K':
			y_bin[i] = 1.0
		if y[i] == ' <=50K':
			y_bin[i] = -1.0
	return y_bin

def predict(w, feature):
	activation = np.dot(w, feature)
	return activation

# Perceptron's
def normalPerceptronTrain(x_train, y_train, w):
	for i in range(0, len(y_train)):
		if y_train[i] * predict(w, x_train[i]) <= 0:
			w += np.multiply(x_train[i], y_train[i])
	return w

def avgPerceptronNaiveTrain(x_train, y_train, w, wPrime, c):
	for i in range(len(y_train)):
		c += 1
		if y_train[i] * predict(w, x_train[i]) <= 0:
			w += np.multiply(x_train[i], y_train[i])
		wPrime += w
	return (wPrime/c), w, wPrime, c

def avgPerceptronSmartTrain(x_train, y_train, w, wPrime, c):
	for i in range(len(y_train)):
		c += 1
		if y_train[i] * predict(w, x_train[i]) <= 0:
			w = w + np.multiply(x_train[i], y_train[i])
			#variable learning rate:
			#w = w + np.dot(x_train[i], y_train[i]/ c)
			wPrime = wPrime + c*np.multiply(x_train[i], y_train[i])
	return (w - (wPrime / float(c))), w, wPrime, c

# MIRA's
def aggr_MIRA(x_train, y_train, w, p):
	for i in range(0, len(y_train)):
		if y_train[i] * predict(w, x_train[i]) <= p:
			w = w + (y_train[i] - np.dot(w, x_train[i])) * x_train[i] / (np.dot(x_train[i], x_train[i]))
	return w

def avg_MIRA(x_train, y_train, w, wPrime, c, p):
	for i in range(len(y_train)):
		c += 1
		if y_train[i] * predict(w, x_train[i]) <= p:
			w = w + (y_train[i] - np.dot(w, x_train[i])) * x_train[i] / (np.dot(x_train[i], x_train[i]))
		wPrime += w
	return (wPrime / float(c)), w, wPrime, c

# Pegasos
def Pegasos_manual(x, y, w, C, flag):
	N = x.shape[0]
	lamda  = 2 / (N * C)
	for i in range(N):
		t = (epoch - 1) * N + i + 1
		eta = 1 / (lamda * t)
		if y[i] * predict(w, x[i]) < 1:
			w -= eta * (lamda * w - np.multiply(x[i], y[i]))
			flag[i] = 1
		else: w -= eta * (lamda * w)
	return w, flag

def primal_objective(w, N, C, x, y):
	counter = 0
	ksi = np.zeros(N)
	for i in range(x.shape[0]):
		ksi[i] = max(0, 1 - np.multiply(y[i], np.dot(w, x[i])))
	c_w = 1 / 2 * np.dot(w, w) + C * sum(ksi)
	return c_w, sum(ksi), np.count_nonzero(ksi)

# Feature engineering

## Squaring the features
'''def my_square(X):
	car_product = itertools.product(X, X)
	return [x * y for x, y in car_product]

def poly_2(X, add_bias = False, interaction_only = False):
	bias = np.ones((X.shape[0], 1))
	if add_bias == True:
		X = np.concatenate((bias, X), axis=1)
	X_new = np.apply_along_axis(my_square, axis = 1, arr = X)
	list_keep = []
	if interaction_only == True:
		for i in range(0, X.shape[1]):
			list_keep.extend(range(i * X.shape[1] + i + 1, (i + 1) * X.shape[1]))
		X_new = X_new[:, tuple(list_keep)]
		X_copy = copy.deepcopy(X_new)
		X_copy = np.concatenate((bias, X_copy), axis=1)
	else:
		for i in range(0, X.shape[1]):
			list_keep.extend(range(i * X.shape[1] + i, (i + 1) * X.shape[1]))
		X_new = (X_new[:, tuple(list_keep)])
		X_copy = copy.deepcopy(X_new)
	return X_copy'''

## Standardize the features
def stand_manual(x):
    x_norm = ((x - x.mean(axis = 0)) / x.std(axis = 0))
    return x_norm


# For answering the questions in HW1 and Ex2
def evaluate(x_valid, y_valid, w):
	accuracy = 0
	totalAccuracy = 0
	numValidation = len(x_valid)
	for i in range(numValidation):
		if y_valid[i] * predict(w, x_valid[i]) > 0:
			accuracy += 1
	totalAccuracy = float(accuracy) / float(numValidation)
	return totalAccuracy

def predicted_positive(x, w):
	num = len(x)
	y_predicted_positive = 0
	for i in range(num):
		if predict(w, x[i]) > 0:
			y_predicted_positive += 1
	return (y_predicted_positive / num)

def predicted_wrong(x, w):
	num = len(x)
	err = []
	for i in range(num):
		if predict(w, x[i]) <= 0:
			err.append(i)
	return err
def predict_batch(x, w):
	y_predicted = []
	for item in x:
		y_predicted.append(np.dot(w, item))
	return y_predicted

## Print out weight vector
def getKey(item):
	return item[1]

## Needs adaptation. But not worring about it since it's not asked for now.
## Probably should make unique_list to be a 1-D array
'''def printWeight(unique_features, w):
	sortedWeightList = []
	for ele, weight in zip(unique_features, w):
		tempList = []
		tempList.append(ele)
		tempList.append(weight)
		sortedWeightList.append(tempList)
	return sorted(sortedWeightList, key=getKey)[::-1]'''

## For shuffling. Should work but slow.
def randomize(X, Y):
    # Generate the permutation index array.
    permutation = np.random.permutation(X.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = X[permutation]
    shuffled_b = Y[permutation]
    return shuffled_a, shuffled_b
def sort_example(X, Y, w):
	X_predicted = np.array(predict_batch(X, w))
	x_y = np.column_stack((X_predicted, Y))
	idx = range(X.shape[0])
	x_y = np.column_stack((idx, x_y))
	x_y = sorted(x_y, key = lambda x: x[-2])
	return np.array(x_y)

# For answering the questions in HW2
def evaluate_SVM(y_real, y_predict):
	accuracy = 0
	totalAccuracy = 0
	n = len(y_real)
	for i in range(n):
		if y_real[i] == y_predict[i]:
			accuracy += 1
	totalAccuracy = float(accuracy) / float(n)
	return 1.0 - totalAccuracy

def predict_positive_SVM(y_predict):
	n = len(y_predict)
	y_predict_positive = 0
	for i in range(n):
		if y_predict[i] > 0:
			y_predict_positive += 1
	return (y_predict_positive / n)



# Read-in data
#trainData = "income.train.txt"
# Read-in the first 5000 cases on the training set
trainData = "income.train.5k.txt"
x_train, num_train, y_train = readData(trainData, num_idx = [0, 7])
validData = "income.dev.txt"
x_valid, num_valid, y_valid = readData(validData, num_idx = [0, 7])
testData = "income.test.txt"
x_test, num_test = readData(testData, test = True, num_idx = [0, 7])

# Sort the data set w.r.t. y
'''train = np.column_stack((x_train, y_train))
train = np.array(sorted(train, key = lambda x: x[-1]))
x_train = train[:, :-1]
y_train = train[:, -1]'''

# Binarize features
x_train_bin, unique_train = binarizeFeature(x_train)
x_valid_bin = binarizeFeature(x_valid, unique_train)
x_test_bin = binarizeFeature(x_test, unique_train)

# Generating labels w.r.t. y
y_train_bin = gen_label(y_train)
y_valid_bin = gen_label(y_valid)


# If numerical features are added:
# First, standardize
'''
num_train = stand_manual(num_train)
num_valid = stand_manual(num_valid)
num_test = stand_manual(num_test)'''

'''num_train = preprocessing.scale(num_train)
num_valid = preprocessing.scale(num_valid)
num_test = preprocessing.scale(num_test)'''

# Then, stack
'''x_train_all = np.column_stack((num_train, x_train_bin))
x_valid_all = np.column_stack((num_valid, x_valid_bin))
x_test_all = np.column_stack((num_test, x_test_bin))'''


# Testing binarized features on sklearn
'''poly = PolynomialFeatures(degree=2, interaction_only=True)
x_train = poly.fit_transform(x_train)
x_valid = poly.fit_transform(x_valid)
x_test = poly.fit_transform(x_test)'''

# I wrote my own fucntions to get the polynomial feature, and finally reduced the time to be 9 mins...

'''start_time = time.clock()
x_train = poly_2(x_train, interaction_only = True)
x_valid = poly_2(x_valid, interaction_only = True)
x_test = poly_2(x_test, interaction_only = True)
end_time = time.clock()
print "Time used on squaring the features is", end_time - start_time'''

# Initiating values before training
batch_size = 1000
EPOCHS = 5
learning_rate = 1.0
Accuracies = np.zeros(EPOCHS * len(range(0, len(x_train_bin), batch_size)))
epoch = np.zeros(EPOCHS * len(range(0, len(x_train_bin), batch_size)))
w_avg = np.zeros(x_train_bin.shape[1])
w = np.zeros(x_train_bin.shape[1])
wPrime = np.zeros(x_train_bin.shape[1])
c = 0
weight_vectors = []

# We need to save the weight vector which achieves the best error rate.
start_time = time.clock()
for e in range(EPOCHS):
	print("Training......")
	print()
	# Chuan: To ensure that x and y shuffle together, we reset the random state after shuffle x. Then, when we shuffle y, it gets the same permutation as x.
	'''rng_state = np.random.get_state()
	np.random.shuffle(x_train_bin)
	np.random.set_state(rng_state)
	np.random.shuffle(y_train_bin)'''

	# Chuan: I don't think we should reset everything to be zero at the beginning of each iteration.
	sampleSize = batch_size
	p = 0.9
	for i, offset in enumerate(range(0, len(x_train_bin), batch_size)):
		batch_x, batch_y = x_train_bin[offset:offset+batch_size], y_train_bin[offset:offset+batch_size]

		# the following the five different peceptron updating scheme
		#w_avg = normalPerceptronTrain(batch_x, batch_y, w_avg)
		#w_avg, w, wPrime, c = avgPerceptronNaiveTrain(batch_x, batch_y, w, wPrime, c)
		w_avg, w, wPrime, c = avgPerceptronSmartTrain(batch_x, batch_y, w, wPrime, c)
		#w_avg = aggr_MIRA(batch_x, batch_y, w_avg, p)
		#w_avg, w, wPrime, c = avg_MIRA(batch_x, batch_y, w, wPrime, c, p)
		weight_vectors.append(w_avg)
		validationAccuracy = evaluate(x_valid_bin, y_valid_bin, w_avg)
		trainAccuracy  = evaluate(x_train_bin, y_train_bin, w_avg)
		Accuracies[e * len(range(0, len(x_train_bin), batch_size)) + i] = validationAccuracy
		epoch[e * len(range(0, len(x_train_bin), batch_size)) + i] = e + sampleSize / float(len(x_train_bin))
		print("EPOCH {:.3f} ...".format(min(e + sampleSize / float(len(x_train_bin)), EPOCHS)))
		print "train error = {:.4f}, ".format(1 - trainAccuracy), "dev error = {:.4f}".format(1 - validationAccuracy)
		print(" ")
		sampleSize += batch_size
end_time = time.clock()
#print "Time used on training using average MIRA is", end_time - start_time
print "Time used on training using average perceptron is", end_time - start_time

#print weight_vectors
weight_vectors = np.array(weight_vectors)
print "Best dev error rate %.4f acheived at epoch %.3f" % (1 - max(Accuracies), min(epoch[np.argmax(Accuracies)], EPOCHS))

# Select the weight vector that gives us the best error rate
w_avg = weight_vectors[np.argmax(Accuracies), :]
unique_train_flat = np.concatenate(unique_train)

'''weightList = printWeight(unique_train_flat, w_avg)
for item in weightList:
	print(item)'''

print "Predicted positive rate on train set is {:.4f}".format(predicted_positive(x_train_bin, w_avg))
print "Predicted positive rate on dev set is {:.4f}".format(predicted_positive(x_valid_bin, w_avg))
print "Predicted positive rate on test set is {:.4f}".format(predicted_positive(x_test_bin, w_avg))
print " "

# HW2: Shangjia's Q1
C = 1
# HW2 -- 1.1
clf = svm.SVC(kernel = 'linear', C = C)
t = time.time()
clf.fit(x_train_bin[:,1:], y_train_bin)
print "training costs %f seconds" %(time.time() - t)

print("training error: {:.5f}".format(1 - clf.score(x_train_bin[:,1:], y_train_bin)))
print("dev error: {:.5f}".format(1- clf.score(x_valid_bin[:,1:], y_valid_bin)))

svmmodel = np.concatenate((clf.coef_[0], clf.intercept_))
# HW2 -- 1.2
# print the number of the suppot vectors
print clf.n_support_
# print number of the margin vilation
prec = 1e-04
sign = lambda x: -1 if x < -prec else 1 if x > prec else 0
num_vio = 0
for ps, alpha in zip(clf.support_vectors_, clf.dual_coef_[0]):
	slack = (1-sign(alpha) * (svmmodel.dot(np.append(ps, 1))))
	if (abs(alpha) == C) and (slack > 0):
		num_vio += 1
print num_vio
# HW2 -- 1.3
total_vio = 0
for ps, alpha in zip(clf.support_vectors_, clf.dual_coef_[0]):
	slack = (1-sign(alpha) * (svmmodel.dot(np.append(ps, 1))))
	if (abs(alpha) == C) and (slack > 0):
		total_vio += slack
print total_vio
print "objective function", 0.5 * np.dot(svmmodel, svmmodel) + C * total_vio
# HW2 -- 1.4
# track the index of support vector and its index in original training set
sv_idx_list = []
for idx, val in enumerate(clf.support_):
	sv_idx_list.append([idx, val])
pos_slack_list = []
neg_slack_list = []
# keep tracking the index of the support vector
for idx, alpha in zip(sv_idx_list, clf.dual_coef_[0]):
	slack = (1-sign(alpha) * (svmmodel.dot(np.append(clf.support_vectors_[idx[0]], 1))))
	if (abs(alpha) == C) and (slack >= 0):
		if y_train_bin[idx[1]] == 1:
			pos_slack_list.append([idx[0], idx[1], slack])
		if y_train_bin[idx[1]] == -1:
			neg_slack_list.append([idx[0], idx[1], slack])
def getKey(item):
	return item[2]
pos_slack_list_sorted = sorted(pos_slack_list, key=getKey)[::-1]
neg_slack_list_sorted = sorted(neg_slack_list, key=getKey)[::-1]
# positive example, top five slack
for item in pos_slack_list_sorted[:5]:
	print x_train[item[1]], item[2]
# negative example, top five slack
for item in neg_slack_list_sorted[:5]:
	print x_train[item[1]], item[2]
# HW2, 1.5
C = [0.01, 0.1, 1, 2, 5, 10]
train_err = []
dev_err = []
for c_instance in C:
	clf = svm.SVC(kernel = 'linear', C = c_instance)
	t = time.time()
	clf.fit(x_train_bin[:, 1:], y_train_bin)
	running_time = time.time() - t
	pred_accuracy_train = clf.score(x_train_bin[:, 1:], y_train_bin)
	pred_accuracy_valid = clf.score(x_valid_bin[:, 1:], y_valid_bin)
	print "training time {:.5f} seconds, training error {:.5f}, dev error {:.5f}, # of support vectors {}".format(running_time, 1-pred_accuracy_train, 1-pred_accuracy_valid, len(clf.support_vectors_))
	train_err.append((1-pred_accuracy_train)*100)
	dev_err.append((1-pred_accuracy_valid)*100)
# HW2, 1.6
# C = 5 has the lowest dev rate
# HW2, 1.7
plt.plot(C, train_err, 'b-*', linewidth = 2.5, label = 'Training error')
plt.plot(C, dev_err, 'r-o', linewidth = 2.5, label = 'Validation error')
plt.subplots_adjust(left=0.12, right=0.90, top=0.9, bottom=0.36)
plt.xlabel('C', fontsize = 15)
plt.ylabel('Error Rate (%)', fontsize = 15)
plt.legend(loc = 1, fontsize = 14)
plt.show()

# HW2, 1.8
x_train_bin_5 = x_train_bin[3:8] # to include both classes
y_train_bin_5 = y_train_bin[3:8]

x_train_bin_50 = x_train_bin[:50]
y_train_bin_50 = y_train_bin[:50]

x_train_bin_500 = x_train_bin[:500]
y_train_bin_500 = y_train_bin[:500]

num_train = [5, 50, 500, 5000]
x_train_set = [x_train_bin_5, x_train_bin_50, x_train_bin_500, x_train_bin]
y_train_set = [y_train_bin_5, y_train_bin_50, y_train_bin_500, y_train_bin]

training_time_list = []
for x_train_instance, y_train_instance in zip(x_train_set, y_train_set):
	clf = svm.SVC(kernel = 'linear', C = 1)
	t = time.time()
	clf.fit(x_train_instance[:,1:], y_train_instance)
	training_time_list.append(time.time() - t)

for n, t in zip(num_train, training_time_list):
	print (t* 1e7 / n) , (t* 1e8 / n**2),  (t* 1e8 / n**3) # check the ratio

plt.plot(num_train, training_time_list, 'b-*', linewidth = 2.5)
plt.subplots_adjust(left=0.12, right=0.90, top=0.9, bottom=0.36)
plt.xlabel('Number of training examples', fontsize = 15)
plt.ylabel('Training time (sec)', fontsize = 15)
plt.legend(loc = 1, fontsize = 14)
plt.title("Training time VS. Number of training examples")
plt.show()

# Basic SVM
##Need to remove bias, and add concept in the functional margin
clf = svm.SVC(kernel='linear', C=1)
start_time = time.clock()
#Remove bias that we added, since sklearn will add that for us automatically
x_train_bin_sk = x_train_bin[:, 1:]
x_valid_bin_sk = x_valid_bin[:, 1:]
x_test_bin_sk = x_test_bin[:, 1:]
clf.fit(x_train_bin_sk, y_train_bin.ravel())
end_time = time.clock()
print "Time used on training using SVM with linear kernel is", end_time - start_time
print "Number of support vector is:", clf.n_support_
y_train_bin_predict = clf.predict(x_train_bin_sk)
y_valid_bin_predict = clf.predict(x_valid_bin_sk)
y_test_bin_predict = clf.predict(x_test_bin_sk)
w = clf.coef_
w =  w.ravel()
b = clf.intercept_
w = np.concatenate((b, w))
N = x_train_bin.shape[0]
C = 1
cw_SVM, sum_ksi_SVM, total_violated_SVM = primal_objective(w, N, C, x_train_bin, y_train_bin)
print "Objective function for SVM with linear kernel is {:.1f}".format(cw_SVM)
print "Total margin violation for SVM with linear kernel is {:.1f}".format(sum_ksi_SVM)
print "Total margin violated observation of SVM with linear kernel is {:d}".format(total_violated_SVM)
print "Train error in SVM with linear kernal is {:.4f}".format(evaluate_SVM(y_train_bin.ravel(), y_train_bin_predict))
print 1 - clf.score(x_train_bin_sk, y_train_bin.ravel())
print "Dev error in SVM with linear kernal is {:.4f}".format(evaluate_SVM(y_valid_bin.ravel(), y_valid_bin_predict))
print 1 - clf.score(x_valid_bin_sk, y_valid_bin.ravel())
print "Predicted positive rate on train set is {:.4f}".format(predict_positive_SVM(y_train_bin_predict))
print "Predicted positive rate on dev set is {:.4f}".format(predict_positive_SVM(y_valid_bin_predict))
print "Predicted positive rate on test set is {:.4f}".format(predict_positive_SVM(y_test_bin_predict))
print " "

# SVM with quadratic kernel
# C = 115 to 128 is the best C I tuned, 265-280 Shangjia tuned
clf_2 = svm.SVC(kernel='poly', degree = 2, coef0 = 1, C = 270)
start_time = time.clock()
clf_2.fit(x_train_bin_sk, y_train_bin.ravel())
end_time = time.clock()
print "Time used on training using SVM with quadratic kernel is", end_time - start_time
y_train_bin_predict_2 = clf_2.predict(x_train_bin_sk)
y_valid_bin_predict_2 = clf_2.predict(x_valid_bin_sk)
y_test_bin_predict_2 = clf_2.predict(x_test_bin_sk)
print "Train error in SVM with quadratic kernal is {:.4f}".format(evaluate_SVM(y_train_bin.ravel(), y_train_bin_predict_2))
print 1 - clf_2.score(x_train_bin_sk, y_train_bin.ravel())
print "Dev error in SVM with quadratic kernal is {:.4f}".format(evaluate_SVM(y_valid_bin.ravel(), y_valid_bin_predict_2))
print 1 - clf_2.score(x_valid_bin_sk, y_valid_bin.ravel())
print "Predicted positive rate on train set is {:.4f}".format(predict_positive_SVM(y_train_bin_predict_2))
print "Predicted positive rate on dev set is {:.4f}".format(predict_positive_SVM(y_valid_bin_predict_2))
print "Predicted positive rate on test set is {:.4f}".format(predict_positive_SVM(y_test_bin_predict_2))
print " "

# Ex2, 3.3
c_list = []
for i in range(1,201):
	c_list.append(5 * i)
quad_training_error_list = []
quad_dev_error_list = []
quad_training_time = []
quad_num_support_vector = []
for c in c_list:
	t3 = time.time()
	clf = svm.SVC(kernel = 'poly', degree = 2, coef0 = 1, C = c)
	clf.fit(x_train_bin[:,1:], y_train_bin)
	quad_training_time.append(time.time() - t3)
	quad_training_error_list.append((1-clf.score(x_train_bin[:,1:], y_train_bin)) * 100)
	quad_dev_error_list.append((1-clf.score(x_valid_bin[:,1:], y_valid_bin)) * 100)
	quad_num_support_vector.append(len(clf.support_vectors_))
plt.plot(c_list[:175], quad_training_error_list, 'b-*', linewidth = 2.5, label = "Training error")
plt.plot(c_list[:175], quad_dev_error_list, 'r-o', linewidth = 2.5, label = "Dev error")
plt.subplots_adjust(left=0.12, right=0.90, top=0.9, bottom=0.36)
plt.xlabel('C', fontsize = 15)
plt.ylabel('Error (%)', fontsize = 15)
plt.title("Quadratic kernel support vector machine")
plt.legend(loc = 1, fontsize = 14)
plt.show()
#===========================================================================================
plt.plot(c_list, quad_training_time, 'g-d', linewidth = 2.5)
plt.subplots_adjust(left=0.12, right=0.90, top=0.9, bottom=0.36)
plt.xlabel('C', fontsize = 15)
plt.ylabel('Training Time (sec)', fontsize = 15)
plt.title("Quadratic kernel support vector machine")
plt.legend(loc = 1, fontsize = 14)
plt.show()

with open("quad.txt", 'w') as f:
	for i in range(len(quad_training_time)):
		f.write(str(quad_training_time[i]) + ' ')
		f.write(str(quad_training_error_list[i]) + ' ')
		f.write(str(quad_dev_error_list[i]) + ' ')
		f.write(str(quad_num_support_vector[i]) +'\n')

with open("svm.txt") as f:
	svm_data = f.read().splitlines()
svm = [map(float, item.split('\t')) for item in svm_data]
svm = np.array(svm)

plt.plot(svm[:,0], np.multiply(svm[:,1], 100), 'b-*', linewidth = 2.5, label = "Training error")
plt.plot(svm[:,0], np.multiply(svm[:,2], 100), 'r-o', linewidth = 2.5, label = "Dev error")
plt.subplots_adjust(left=0.12, right=0.90, top=0.9, bottom=0.36)
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Error (%)', fontsize = 15)
#plt.title("")
plt.legend(loc = 1, fontsize = 14)
plt.show()

#===========================================================
plt.plot(svm[:,0], svm[:,3], 'b-*', linewidth = 2.5)
plt.subplots_adjust(left=0.12, right=0.90, top=0.9, bottom=0.36)
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Objective function', fontsize = 15)
#plt.title("")
plt.legend(loc = 1, fontsize = 14)
plt.show()


# Pegasos
w = np.zeros(x_train_bin.shape[1])
t = 1
C = 1
train_error = []
dev_error = []
trainAccuracy_old = 0
validAccuracy_old = 0
cw_old = 0
cw_new = 100000000
trainAccuracy_new = 1
validAccuracy_new = 1
epoch = 1
N = x_train_bin.shape[0]
lda = 2 / (N * C)
flag = np.zeros(N)
epoch_list = []
train_error_list = []
dev_error_list = []
obj_list = []

start_time = time.clock()
while abs(cw_old - cw_new) > 0.5 or abs(validAccuracy_old - validAccuracy_new) > 0.0001:
	#print("Training Pegasos, epoch = {:d}...".format(epoch))
	trainAccuracy_old = trainAccuracy_new
	validAccuracy_old = validAccuracy_new
	cw_old = cw_new
	#print cw_old
	# Chuan: To ensure that x and y shuffle together, we reset the random state after shuffle x. Then, when we shuffle y, it gets the same permutation as x.
	'''rng_state = np.random.get_state()
	np.random.shuffle(x_train_bin)
	np.random.set_state(rng_state)
	np.random.shuffle(y_train_bin)'''
	epoch_list.append(epoch)
	w, flag = Pegasos_manual(x_train_bin, y_train_bin, w, C, flag)
	cw_new, sum_ksi_new, total_violated_new = primal_objective(w, N, C, x_train_bin, y_train_bin)
	#print cw_new
	#print sum_ksi_new
	#print total_violated_new
	trainAccuracy_new = evaluate(x_train_bin, y_train_bin, w)
	validAccuracy_new = evaluate(x_valid_bin, y_valid_bin, w)
	train_error_list.append(1 - trainAccuracy_new)
	dev_error_list.append(1 - validAccuracy_new)
	obj_list.append(cw_new)
	#print "Objective function = {:d}".format(int(cw_new)), "train error = {:.4f}, ".format(1 - trainAccuracy_new), "dev error = {:.4f}".format(1 - validAccuracy_new)
	#print(" ")
	epoch += 1
end_time = time.clock()
print "Time used on training using Pegasos with linear kernel is", end_time - start_time
print "Number of epochs needed for training: {:d}".format(epoch)

num_update = np.count_nonzero(flag)
print "Number of support vector for Pegasos with linear kernel is:", num_update
print "Objective function for Pegasos with linear kernel is {:.1f}".format(cw_new)
print "Total margin violation for Pegasos with linear kernel is {:.1f}".format(sum_ksi_new)
print "Total margin violated observation for Pegasos with linear kernel is {:d}".format(total_violated_new)
print "Objective function = {:d}".format(int(cw_new)), "train error = {:.4f}, ".format(1 - trainAccuracy_new), "dev error = {:.4f}".format(1 - validAccuracy_new)
print(" ")

epoch_list = np.array(epoch_list)
train_error_list = np.array(train_error_list)
dev_error_list = np.array(dev_error_list)
obj_list = np.array(obj_list)
A = np.column_stack((epoch_list, train_error_list, dev_error_list, obj_list))
#np.savetxt("SVM.csv", A, delimiter = ",")


'''
plt.plot(epoch_list, obj_list, 'b-*', linewidth = 2.5)
plt.subplots_adjust(left=0.12, right=0.90, top=0.9, bottom=0.36)
plt.xlabel('Number of epochs', fontsize = 15)
plt.ylabel('Objective function', fontsize = 15)
plt.legend(loc = 1, fontsize = 14)
plt.title("Objective Fuction Vs. Number of Epochs")
#fig = plt.figure()
#plt.tight_layout()
#fig.savefig("plot.png")
plt.show()
plt.plot(epoch_list, train_error_list, 'b-*', linewidth = 2.5, label = "Training error")
plt.plot(epoch_list, dev_error_list, 'r-o', linewidth = 2.5, label = "Dev error")
plt.subplots_adjust(left=0.12, right=0.90, top=0.9, bottom=0.36)
plt.xlabel('Number of epochs', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend(loc = 1, fontsize = 14)
plt.title("Train/Dev Error vs. Number of Epochs")
plt.show()'''


'''print(sort_example(x_valid, y_valid, w_avg))[:5, :]
print(sort_example(x_valid, y_valid, w_avg)[-5:, :])
neg = sort_example(x_valid, y_valid, w_avg)[:5, 0]
pos = sort_example(x_valid, y_valid, w_avg)[-5:, 0]
for i in range(5):
	print valid[int(pos[i]), :]
for i in range(5):
	print valid[int(neg[i]), :]
errs = np.array(predicted_wrong(x_valid, w_avg))
print errs.shape[0]
five_errs = errs[-5:]
for i in range(5):
	print valid[int(five_errs[i]), :]'''

'''errorRate = np.array(errorRate)
plt.plot(errorRate[:,0], errorRate[:,1] * 100, 'b-*', linewidth = 2.5, label = 'Average MIRA')
plt.subplots_adjust(left=0.12, right=0.90, top=0.9, bottom=0.36)
plt.axis([1, 5, 0, 100])
plt.xlabel('EPOCHS', fontsize = 15)
plt.ylabel('Error Rate (%)', fontsize = 15)
plt.title('MIRA Training Performance, p = 0.9', fontsize = 18)
plt.legend(loc = 1, fontsize = 14)
#plt.tight_layout()
plt.show()'''

# Writing predicted labels in test data.
testData ="income.test.txt"

predicted_target = [">50k" if item == 1 else "<=50k" for item in y_test_bin_predict_2]
with open(testData) as f:
	incomeData = f.read().splitlines()

with open("income.test.predicted.txt", "w") as f:
	for feature, target in zip(incomeData, predicted_target):
		f.write(str(feature) + ' ' + str(target) + '\n')
