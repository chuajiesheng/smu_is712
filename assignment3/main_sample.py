import numpy as np
from sklearn import svm
from sklearn import linear_model
import random
import sys
from scipy import sparse
import time

def feature_norm(X):
  	mu = np.mean(X, 0)
  	sigma = np.std(X, 0)
  	X_norm = (X - mu) / sigma
	return X_norm

def importdata(path):
	fpr = open(path, 'r')
	X = []
	for line in fpr:
		vals = line.strip().split(' ')
		fea = [float(val) for val in vals]
		X.append(fea)
	fpr.close()
	return X

def data_split(data_name, sparse_bool):
	if data_name == 'heart':
		random.seed(123)
		X = importdata("../data/heart.dat")
		random.shuffle(X)
		X = np.array(X)
		Y = X[:,-1] - 1
		X = X[:,0:-1]
		X = feature_norm(X)

		data_size = X.shape[0]
		train_size = np.floor(data_size / 10) * 7
		val_size = np.floor(data_size / 10)
		test_size = data_size - train_size - val_size

		X_train = X[0:train_size]
		Y_train = Y[0:train_size]
		X_val   = X[train_size:train_size+val_size]
		Y_val   = Y[train_size:train_size+val_size]
		X_test  = X[train_size+val_size:]
		Y_test  = Y[train_size+val_size:]

	elif data_name == 'gisette':
		X_train = importdata("../data/gisette_train.data")
		X_train = np.array(X_train) / 1000
		Y_train = importdata("../data/gisette_train.labels")
		Y_train = ( np.array(Y_train)[:,0] + 1 ) / 2

		X = importdata("../data/gisette_valid.data")
		X = np.array(X) / 1000;
		Y = importdata("../data/gisette_valid.labels")
		Y = ( np.array(Y)[:,0] + 1) / 2
		
		val_num = np.floor(X.shape[0]/2);
		X_val = X[0:val_num]
		Y_val = Y[0:val_num]
		X_test = X[val_num:]
		Y_test = Y[val_num:]
	else:
		print 'no existing data!!!'
	
	#Your code here
	if sparse_bool == 'sparse':
		X_train = 
		X_val   = 
		X_test  = 
	return X_train, Y_train, X_val, Y_val, X_test, Y_test

#Your code here
def model_test(X_test, Y_test, svr):
	
	accuracy = 
	
	return accuracy

#Your code here
def model_train(X_train, Y_train, model_name):
	
	if model_name == 'svm_kernel':
		svr = 	

	elif model_name == 'svm_linear':
		svr = 	

	elif model_name == 'logisticR':
		svr = 	

	else:
		print 'no existing model!!'
		
	
	return svr

def main():
	
	#command: python main.py heart svm_linear sparce
	data_name = sys.argv[1] # heart or gisette
	model_name = sys.argv[2] # svm_linear or logisticR or svm_kernel
	sparse_bool = sys.argv[3] # sparse or not

	X_train, Y_train, X_val, Y_val, X_test, Y_test = data_split(data_name, sparse_bool)
	
	start_time = time.time()
	svr = model_train(X_train, Y_train, model_name)
	print("training time: %s seconds" % (time.time() - start_time))
	
	accuracy_train = model_test(X_train, Y_train, svr)
	accuracy_val = model_test(X_val, Y_val, svr)
	accuracy_test = model_test(X_test, Y_test, svr)
	print ("training data set accuracy:" + str(accuracy_train))
	print ("validiction data set accuracy:" + str(accuracy_val))
	print ("test data set accuracy:" + str(accuracy_test))

if __name__ == '__main__':
	main()
