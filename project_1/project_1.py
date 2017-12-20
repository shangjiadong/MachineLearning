from __future__ import division
import numpy as np
import scipy
import os
import itertools
import operator
import time
import timeit
from numpy import linalg
from sklearn.preprocessing import PolynomialFeatures

os.chdir("C:/MachineLearning/pj1/pj1-data")

# read data
def readData(file):
	"""
	This function takes the file name as the input, and returns the processed
	feature array.
	I swapped the sequence of the Hour and Workclass, so later if you need to
	focus on the integer feature, you can just not process income[:,0] and income[:,1]
	for example:
		income[:,0] = [int(item) for item in income[:,0]]
		income[:,1] = [int(item) for item in income[:,7]]
	just use them to replace the income[:,0] and income[:,1] below.
	"""
	with open(file) as f:
		incomeData = f.read().splitlines()
	income = [map(str, item.split(', ')) for item in incomeData]
	income = np.array(income)
	num_age = np.copy(income[:, 0]).astype(int)
	num_hour = np.copy(income[:, 7]).astype(int)
	tempFeature = np.copy(income[:,1])
	income[:,0] = ["Age: "+str(item) for item in income[:,0]]
	income[:,1] = ['Hour: '+str(item) for item in income[:,7]]
	income[:,2] = ["Education: "+str(item) for item in income[:,2]]
	income[:,3] = ["MaritalStatus: "+str(item) for item in income[:,3]]
	income[:,4] = ["Occupation: "+str(item) for item in income[:,4]]
	income[:,5] = ["Race: "+str(item) for item in income[:,5]]
	income[:,6] = ["Sex: "+str(item) for item in income[:,6]]
	income[:,7] = ["WorkClass: "+str(item) for item in tempFeature]
	income[:,8] = ['Country: '+str(item) for item in income[:,8]]
	income[:,9] = ['Target: '+str(item) for item in income[:,9]]
	# Uneable bias temporarily inorder to test on sklearn
	bias = np.array([[1] * income.shape[0]])
	income = np.concatenate((bias.T, income), axis = 1)
	return income, num_age, num_hour

def readtest(file):
	"""
	This function takes the file name as the input, and returns the processed
	feature array.
	I swapped the sequence of the Hour and Workclass, so later if you need to
	focus on the integer feature, you can just not process income[:,0] and income[:,1]
	for example:
		income[:,0] = [int(item) for item in income[:,0]]
		income[:,1] = [int(item) for item in income[:,7]]
	just use them to replace the income[:,0] and income[:,1] below.
	"""
	with open(file) as f:
		incomeData = f.read().splitlines()
	income = [map(str, item.split(', ')) for item in incomeData]
	income = np.array(income)
	num_age = np.copy(income[:, 0]).astype(int)
	num_hour = np.copy(income[:, 7]).astype(int)
	tempFeature = np.copy(income[:,1])
	income[:,0] = ["Age: "+str(item) for item in income[:,0]]
	income[:,1] = ['Hour: '+str(item) for item in income[:,7]]
	income[:,2] = ["Education: "+str(item) for item in income[:,2]]
	income[:,3] = ["MaritalStatus: "+str(item) for item in income[:,3]]
	income[:,4] = ["Occupation: "+str(item) for item in income[:,4]]
	income[:,5] = ["Race: "+str(item) for item in income[:,5]]
	income[:,6] = ["Sex: "+str(item) for item in income[:,6]]
	income[:,7] = ["WorkClass: "+str(item) for item in tempFeature]
	income[:,8] = ['Country: '+str(item) for item in income[:,8]]
	#income[:,9] = ['Target: '+str(item) for item in income[:,9]]
	#target = np.array([[0] * income.shape[0]])
	#income = np.concatenate((income, target.T), axis = 1)
	# Uneable bias temporarily inorder to test on sklearn
	bias = np.array([[1] * income.shape[0]])
	income = np.concatenate((bias.T, income), axis = 1)
	Target = np.copy(income[:, 1])
	for i in range(income.shape[0]):
		if i % 2 == 0:
			Target[i] = 'Target: >50K'
		else:
			Target[i] = 'Target: <=50K'
	Target = np.asarray(Target)
	income = np.column_stack((income, Target.T))
	return income, num_age, num_hour

def binarizeFeature(train, uniTrainEle):
	"""
	This function takes training dataset, and the uniTrainingElement arrary as
	the input, and returns the binarized the feature dataset.
	Be careful, here we binarize all the features, if you need to do something
	about the Age and Hour (there are 166 unique elements, plus 1 for bias), do not
	binarize the first two column. Basically you just need to binarize the rest, and
	concatenate the Age and Hour column back to the array, like I did in readData().
	The following code can binarize the rest and keep the two column integer

	trainFeature = np.zeros((len(train), len(uniTrainEle)-167))
	for k in range(0, len(train)):
		for i in range(3,11):
			for j in range(167, len(uniTrainEle)):
				if train[k][i] == uniTrainEle[j]:
					trainFeature[k][j-167] = 1.0
	Hour = np.array([train[:,2]])
	trainFeature = np.concatenate((Hour.T, trainFeature), axis = 1)
	Age = np.array([train[:,1]])
	trainFeature = np.concatenate((Age.T, trainFeature), axis = 1)
	return trainFeature

	"""
	trainFeature = np.zeros((len(train), len(uniTrainEle)))
	for k in range(0, len(train)):
		for i in range(len(train[k])):
			for j in range(len(uniTrainEle)):
				if train[k][i] == uniTrainEle[j]:
					trainFeature[k][j] = 1.0
	return trainFeature
def predict(w, feature):
	activation = np.dot(w, feature)
	return activation

# Perceptron's
def normalPerceptronTrain(x_train, y_train, w):
	for i in range(0, len(y_train)):
		if y_train[i] * predict(w, x_train[i]) <= 0:
			w += np.dot(x_train[i], y_train[i])
	return w

def avgPerceptronNaiveTrain(x_train, y_train, w, wPrime, c):
	for i in range(len(y_train)):
		c += 1
		if y_train[i] * predict(w, x_train[i]) <= 0:
			w += np.dot(x_train[i], y_train[i])
		wPrime += w
	return (wPrime/c), w, wPrime, c

def avgPerceptronSmartTrain(x_train, y_train, w, wPrime, c):
	for i in range(len(y_train)):
		c += 1
		if y_train[i] * predict(w, x_train[i]) <= 0:
			#w = w + np.dot(x_train[i], y_train[i])
			#variable learning rate:
			w = w + np.dot(x_train[i], y_train[i]/ c)
			wPrime = wPrime + c*np.dot(x_train[i], y_train[i])
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

def findUnique(train):
	dataDiscription = []
	for i in range(train.shape[1]):
		dataDic = {}
		for item in train[:,i]:
			if item in dataDic:
				dataDic[item] += 1
			else:
				dataDic[item] = 1
		dataDiscription.append(dataDic)
	return dataDiscription

# Print out weight vector
def getKey(item):
	return item[1]

def printWeight(uniTrainEle, w):
	sortedWeightList = []
	for ele, weight in zip(uniTrainEle[:-2], w):
		tempList = []
		tempList.append(ele)
		tempList.append(weight)
		sortedWeightList.append(tempList)
	return sorted(sortedWeightList, key=getKey)[::-1]

# Feature engineering
def my_square(X):
	car_product = itertools.product(X, X)
	return [x * y for x, y in car_product]

def poly_2(X, add_bias = False, interaction_only = False):
	if add_bias == True:
		bias = np.ones((X.shape[0], 1))
		X = np.concatenate((bias, X), axis=1)
	X_new = np.apply_along_axis(my_square, axis = 1, arr = X)
	if interaction_only == True:
		for i in range(1, X.shape[1]):
			X_new = np.delete(X_new, i * X.shape[1] + 1, 1)
	return X_new

# Looking for 'interaction only = True' option. Note: Bias should be already added to the X matrix. O.w. add_bias should = True. But too slow.
'''def poly_2(X, interaction_only = False, add_bias = False):
	if add_bias == True:
		bias = np.ones((X.shape[0], 1))
		X = np.concatenate((bias, X), axis=1)
	X_new = np.empty((X.shape[0], 0))
	if interaction_only == True:
		for i in range(X.shape[1]):
			for j in range(i + 1, X.shape[1]):
				#X_new = np.column_stack((X_new, X[:, i] * X[:, j]))
				X_new = np.column_stack((X_new, list(map(operator.mul, X[:, i], X[:, j]))))
		bias = np.ones((X_new.shape[0], 1))
		X_new = np.concatenate((bias, X_new), axis=1)
		return X_new
	else:
		for i in range(X.shape[1]):
			for j in range(i, X.shape[1]):
				#X_new = np.column_stack((X_new, X[:, i] * X[:, j]))
				X_new = np.column_stack((X_new, list(map(operator.mul, X[:, i], X[:, j]))))
		return X_new'''

def stand_manual(x):
    x_norm = ((x - x.mean(axis = 0)) / x.std(axis = 0))
    return x_norm

trainData = "income.train.txt"
train, age_train, hour_train = readData(trainData)

numTrainExamples = train.shape[0]
numTrainFeatures = train.shape[1]

# validation data
validData = "income.dev.txt"
valid, age_valid, hour_valid = readData(validData)
numValidExamples = valid.shape[0]
numValidFeatures = valid.shape[1]

# test data
#testData ="income.test.txt"
#test, age_test, hour_test= readtest(testData)
testData ="predicted_target.txt"
test, age_test, hour_test= readData(testData)

numTestExamples = test.shape[0]
numTestFeatures = test.shape[1]

batch_size = 1000
EPOCHS = 5
learning_rate = 1.0
# binarize the training data
uniTrainEle = []
for i in range(0, train.shape[1]):
	uniTrainEle.extend(np.unique(train[:,i]))

# Sort the data set w.r.t. y
#train = sorted(train, key = lambda x: x[-1])

trainFeature = binarizeFeature(train, uniTrainEle)
numBinTrainFeatures = trainFeature.shape[1]
uniValidEle = uniTrainEle[:] # validation has to keep consistent with training data
validFeature = binarizeFeature(valid, uniValidEle)
numBinValidFeatures = validFeature.shape[1]
uniTestEle = uniTrainEle[:]
testFeature = binarizeFeature(test, uniTestEle)
numBinTestFeatures = testFeature.shape[1]

x_train = trainFeature[:, 0:(numBinTrainFeatures-2)]
x_valid = validFeature[:, 0:(numBinValidFeatures-2)]
x_test = testFeature[:, 0:(numBinValidFeatures-2)]

# If numerical features are added:
# First, standardize
num_train = np.array(np.column_stack((hour_train, age_train)), dtype = float)
num_valid = np.array(np.column_stack((hour_valid, age_valid)), dtype = float)
num_test = np.array(np.column_stack((hour_test, age_test)), dtype = float)
num_train = stand_manual(num_train)
num_valid = stand_manual(num_valid)
num_test = stand_manual(num_test)

'''num_train = preprocessing.scale(num_train)
num_valid = preprocessing.scale(num_valid)
num_test = preprocessing.scale(num_test)'''

# Then, stack
x_train = np.column_stack((num_train, x_train))
x_valid = np.column_stack((num_valid, x_valid))
x_test = np.column_stack((num_test, x_test))

y_train = np.zeros(trainFeature.shape[0])
for i in range(0, len(train)):
	if train[i][-1] == 'Target: >50K':
		y_train[i] = 1.0
	if train[i][-1] == 'Target: <=50K':
		y_train[i] = -1.0

# Testing binarized features on sklearn
'''poly = PolynomialFeatures(degree=2, interaction_only=True)
x_train = poly.fit_transform(x_train)
x_valid = poly.fit_transform(x_valid)
x_test = poly.fit_transform(x_test)'''

# I wrote my own fucntions to get the polynomial feature, but they take forever to run...

start_time = time.clock()
x_train = poly_2(x_train, interaction_only = True)
x_valid = poly_2(x_valid, interaction_only = True)
x_test = poly_2(x_test, interaction_only = True)
end_time = time.clock()
print "Time used on squaring the features is", end_time - start_time

y_valid = np.zeros(validFeature.shape[0])
for i in range(0, len(valid)):
	if valid[i][-1] == 'Target: >50K':
		y_valid[i] = 1.0
	if valid[i][-1] == 'Target: <=50K':
		y_valid[i] = -1.0

Accuracies = np.zeros(EPOCHS * len(range(0, len(x_train), batch_size)))
epoch = np.zeros(EPOCHS * len(range(0, len(x_train), batch_size)))
w_avg = np.zeros(x_train.shape[1])
w = np.zeros(x_train.shape[1])
wPrime = np.zeros(x_train.shape[1])
c = 0

for e in range(EPOCHS):
	print("Training......")
	print()
	# Chuan: To ensure that x and y shuffle together, we reset the random state after shuffle x. Then, when we shuffle y, it gets the same permutation as x.
	rng_state = np.random.get_state()
	np.random.shuffle(x_train)
	np.random.set_state(rng_state)
	np.random.shuffle(y_train)

	# Chuan: I don't think we should reset everything to be zero at the beginning of each iteration. Move to out of iterations.
	##w = np.zeros(x_train.shape[1])
	##wPrime = np.zeros(x_train.shape[1])
	##c = 0
	sampleSize = 0
	p = 0.9
	for i, offset in enumerate(range(0, len(x_train), batch_size)):
		batch_x, batch_y = x_train[offset:offset+batch_size], y_train[offset:offset+batch_size]

		# the following the five different peceptron updating scheme
		#w_avg = normalPerceptronTrain(batch_x, batch_y, w_avg)
		#w_avg, w, wPrime, c = avgPerceptronNaiveTrain(batch_x, batch_y, w, wPrime, c)
		#w_avg, w, wPrime, c = avgPerceptronSmartTrain(batch_x, batch_y, w, wPrime, c)
		#w_avg = aggr_MIRA(batch_x, batch_y, w_avg, p)
		w_avg, w, wPrime, c = avg_MIRA(batch_x, batch_y, w, wPrime, c, p)
		validationAccuracy = evaluate(x_valid, y_valid, w_avg)
		Accuracies[e * len(range(0, len(x_train), batch_size)) + i] = validationAccuracy
		epoch[e * len(range(0, len(x_train), batch_size)) + i] = e + sampleSize / float(len(x_train))
		##print("EPOCH {:.3f} ...".format(min(e +sampleSize/float(len(x_train)), EPOCHS)))
		print("EPOCH {:.3f} ...".format(min(e + sampleSize / float(len(x_train)), EPOCHS)))
		print("Validation Accuracy = {:.5f}".format(validationAccuracy))
		print(" ")
		sampleSize += batch_size
print "Maximum accuracy %.5f acheived at epoch %.3f" % (max(Accuracies), min(epoch[np.argmax(Accuracies)], EPOCHS))
'''weightList = printWeight(uniTrainEle, w_avg)
for item in weightList:
	print(item)'''
print "Predicted positive rate on train set is {:.4f}".format(predicted_positive(x_train, w_avg))
print "Predicted positive rate on dev set is {:.4f}".format(predicted_positive(x_valid, w_avg))
print "Predicted positive rate on test set is {:.4f}".format(predicted_positive(x_test, w_avg))

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

# test data
testData ="income.test.txt"

predicted_target = [">50k" if predict(item, w_avg) >= 0 else "<=50k" for item in x_test]
'''with open("trained_weight.txt", "w") as f:
	for item in weightList:
		f.write(str(item) + '\n')'''
with open(testData) as f:
	incomeData = f.read().splitlines()

with open("predicted_target_clean.txt", "w") as f:
	for feature, target in zip(incomeData, predicted_target):
		f.write(str(feature) + ' ' + str(target) + '\n')


