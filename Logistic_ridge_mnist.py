# -*- coding: utf-8 -*-


from mnist import MNIST
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import timeit

np.random.seed(42)

def find_27(array):
    res = []
    for i in range(len(array)):
        if array[i] in [2,7]:
            res.append(i)
    return res

def process_label(Y):
    res = np.zeros((len(Y)))
    for i in range(len(Y)):
        if Y[i] == 7:
            res[i] = 1
        else:
            res[i] = -1
    return res

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    
    train_pos = find_27(labels_train)
    X_train = X_train[train_pos,:]
    labels_train = process_label(labels_train[train_pos])
    
    test_pos = find_27(labels_test)
    X_test = X_test[test_pos,:]
    labels_test = process_label(labels_test[test_pos])
    
    X_train = X_train/255.0
    X_test = X_test/255.0
    
    return X_train, labels_train, X_test, labels_test

X_train, labels_train, X_test, labels_test = load_dataset()


class LogisticRidge:
    def __init__(self, lamb,X_test, Y_test, eps=0.00001, optim = 'BGD', learning_rate = 0.01
                 , batch_size = 100):
        self.lamb = lamb
        self.eps = eps
        self.batch_size = 100

        allowed_optim = ['BGD', 'SGD', 'SBGD', 'NEWTON']
        
        if optim.upper() in  allowed_optim:
            self.optim = optim.upper()
        else:
            raise ValueError('Unknown optimizer')
            
        self.learning_rate = learning_rate
        
        self.w = None
        self.b = None
        self.X_test = X_test
        self.Y_test = Y_test
        
        print('Logistic ridge regression initiated with lambda:', lamb,
              ', epsilon:', eps,
              ', optimizer:', optim.upper(),
              ', learning rate:', learning_rate)
        
        
    def loss(self, X, Y, w, b):
        p1 = np.mean(np.log(1+np.exp((-Y *(b + np.matmul(X,w)) ))))
        p2 = self.lamb * np.matmul(w.T, w)
        return p1 + p2
    
    def _gradient(self, X, Y):
        p0 = (1-(1/(1 + np.exp(-Y * (self.b + np.matmul(X, self.w)))))) * (-Y)
        tmp = np.multiply(p0[:,None], X)
        p1 = np.mean(tmp, axis=0   ).T
        p2 = 2* self.lamb * self.w
        w = p1 + p2
        b = np.mean(p0)
        
        return w, b
    
    def _hessian(self, X, Y):
        p0 = (1-(1/(1 + np.exp(-Y * (self.b + np.matmul(X, self.w))))))*\
            np.power(Y,2)* (1/(1 + np.exp(-Y * (self.b + np.matmul(X, self.w)))))
        
        p1 = X.T.dot(np.multiply(p0[:,None], X)) / X.shape[0]
        
        w = p1 + 2*self.lamb*np.identity(X.shape[1])
        
        b = np.mean(p0)
        return w, b
    
    def predict(self,X):
        return np.sign(X.dot(self.w) + self.b)
        
    def get_error(self,X,Y):
        fitted = self.predict(X)
        return np.sum( fitted != Y) / X.shape[0]
        
    def train(self, X, Y, provided = False, w = None, b = None):
        if provided:
            self.w = w
            self.b = b
        else:
            self.w = np.zeros((X.shape[1]))
            self.b = 0
            
        if self.optim == 'BGD':
            res,res1 = self._train_BGD(X,Y)
        elif self.optim == 'SGD':
            self.batch_size = 1
            res,res1 =  self._train_SBGD(X,Y)
        elif self.optim == 'SBGD':
            res,res1 =  self._train_SBGD(X,Y)
        elif self.optim == 'NEWTON':
            res,res1 = self._train_NEWTON(X,Y)
            
        plt.plot(res['Step'], res['TrainLoss'])
        plt.plot(res['Step'], res['TestLoss'])
        plt.title('Step vs training loss and testing loss for ' + self.optim)
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.show()
        
        plt.plot(res1['Step'], res1['TrainError'])
        plt.plot(res1['Step'], res1['TestError'])
        plt.title('Step vs training error and testing error for ' + self.optim)
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Error rate')
        plt.show()
        
        return            
    def _train_BGD(self, X, Y):
        res = pd.DataFrame()
        res1 = pd.DataFrame()
        current_loss = self.loss(X, Y, self.w, self.b)
        print('Training started, initial loss:', current_loss)
        start_time = timeit.default_timer()
        
        
        gradient_w, gradient_b = self._gradient(X, Y)
        new_w = self.w - self.learning_rate * gradient_w
        new_b = self.b - self.learning_rate * gradient_b
        
        new_loss = self.loss(X, Y, new_w, new_b)
        
        step = 1
        while abs(current_loss - new_loss) > self.eps:
            
            
            if step % 10 == 0:
                test_loss = self.loss(self.X_test, self.Y_test, self.w, self.b)
                timeused = timeit.default_timer()-start_time
                print('Step:', step, ', current loss:', current_loss,
                      ', time per step:', timeused/step)
                entry = np.array([[step],[current_loss], [test_loss]]).reshape((1,3))
                res = res.append(pd.DataFrame(entry))
                current_error = self.get_error(X, Y)
                test_error = self.get_error(self.X_test, self.Y_test)
                entry = np.array([[step],[current_error], [test_error]]).reshape((1,3))
                res1 = res1.append(pd.DataFrame(entry))
            
            current_loss = new_loss
            self.w = new_w
            self.b = new_b
            
            gradient_w, gradient_b = self._gradient(X, Y)
            new_w = self.w - self.learning_rate * gradient_w
            new_b = self.b - self.learning_rate * gradient_b
            
            new_loss = self.loss(X, Y, new_w, new_b)
            
            
            
            
            step += 1
            
        print('Training complete. Time used:', timeit.default_timer()-start_time)
        
        res.columns = ['Step', 'TrainLoss', 'TestLoss']
        res1.columns = ['Step', 'TrainError', 'TestError']
        return res,res1
    
    def _train_SBGD(self, X, Y):
        res = pd.DataFrame()
        res1 = pd.DataFrame()
        # scale lambda
        self.lamb = self.lamb / X.shape[0] * self.batch_size
        current_loss = self.loss(X, Y, self.w, self.b)
        print('Training started, initial loss:', current_loss)
        start_time = timeit.default_timer()
        
        
        
        random_index = np.random.choice(X.shape[0], self.batch_size)
        gradient_w, gradient_b = self._gradient(X[random_index,:], Y[random_index])
        new_w = self.w - self.learning_rate * gradient_w
        new_b = self.b - self.learning_rate * gradient_b
        
        new_loss = self.loss(X, Y, new_w, new_b)
        
        step = 1
        non_change = 0
        while True:
            
            if abs(current_loss - new_loss) < self.eps:
                non_change += 1
            else:
                non_change = 0
                
            if non_change > 100:
                break
            
            
            
            
            if step % 10 == 0:
                test_loss = self.loss(self.X_test, self.Y_test, self.w, self.b)
                timeused = timeit.default_timer()-start_time
                print('Step:', step, ', current loss:', current_loss,
                      ', time per step:', timeused/step)
                entry = np.array([[step],[current_loss], [test_loss]]).reshape((1,3))
                res = res.append(pd.DataFrame(entry))
                current_error = self.get_error(X, Y)
                test_error = self.get_error(self.X_test, self.Y_test)
                entry = np.array([[step],[current_error], [test_error]]).reshape((1,3))
                res1 = res1.append(pd.DataFrame(entry))
            
            current_loss = new_loss
            self.w = new_w
            self.b = new_b
            
            random_index = np.random.choice(X.shape[0], 1)
            gradient_w, gradient_b = self._gradient(X[random_index,:], Y[random_index])
            new_w = self.w - self.learning_rate * gradient_w
            new_b = self.b - self.learning_rate * gradient_b
            
            new_loss = self.loss(X, Y, new_w, new_b)
            
            step += 1
            
        print('Training completed. Final loss:', current_loss,
              '. Time used:', timeit.default_timer()-start_time)
        
        res.columns = ['Step', 'TrainLoss', 'TestLoss']
        res1.columns = ['Step', 'TrainError', 'TestError']
        return res,res1
    
    def _train_NEWTON(self, X, Y):
        res = pd.DataFrame()
        res1 = pd.DataFrame()
        current_loss = self.loss(X, Y, self.w, self.b)
        print('Training started, initial loss:', current_loss)
        start_time = timeit.default_timer()
        
        
        
        gradient_w, gradient_b = self._gradient(X, Y)
        hessian_w, hessian_b = self._hessian(X,Y)
        
        new_w = self.w - self.learning_rate * np.linalg.inv(hessian_w).dot(gradient_w)
        new_b = self.b - self.learning_rate * gradient_b / hessian_b
        
        new_loss = self.loss(X, Y, new_w, new_b)
        
        step = 1
        while abs(current_loss - new_loss) > self.eps:
            
            if step % 1 == 0:
                test_loss = self.loss(self.X_test, self.Y_test, self.w, self.b)
                timeused = timeit.default_timer()-start_time
                print('Step:', step, ', current loss:', current_loss,
                      ', time per step:', timeused/step)
                entry = np.array([[step],[current_loss], [test_loss]]).reshape((1,3))
                res = res.append(pd.DataFrame(entry))
                current_error = self.get_error(X, Y)
                test_error = self.get_error(self.X_test, self.Y_test)
                entry = np.array([[step],[current_error], [test_error]]).reshape((1,3))
                res1 = res1.append(pd.DataFrame(entry))
            
            current_loss = new_loss
            self.w = new_w
            self.b = new_b
            
            gradient_w, gradient_b = self._gradient(X, Y)
            hessian_w, hessian_b = self._hessian(X,Y)
            
            new_w = self.w - self.learning_rate * np.linalg.inv(hessian_w).dot(gradient_w)
            new_b = self.b - self.learning_rate * gradient_b / hessian_b
            
            new_loss = self.loss(X, Y, new_w, new_b)
            
            step += 1
            
        print('Training complete. Time used:', timeit.default_timer()-start_time)
        
        res.columns = ['Step', 'TrainLoss', 'TestLoss']
        res1.columns = ['Step', 'TrainError', 'TestError']
        return res,res1
    
#solver_bgd = LogisticRidge(0.1, optim='BGD', X_test=X_test, Y_test=labels_test)
#solver_bgd.train(X_train, labels_train)
        
#solver_sgd = LogisticRidge(0.1, optim='SGD', learning_rate=0.01, eps=0.0001,
#                           X_test=X_test, Y_test=labels_test)
#solver_sgd.train(X_train, labels_train)
#
#
#solver_sbgd = LogisticRidge(0.1, optim='SBGD', learning_rate=0.01, eps=0.0001,
#                            batch_size = 100, X_test=X_test, Y_test=labels_test)
#solver_sbgd.train(X_train, labels_train)
        
    
solver_newton = LogisticRidge(0.1, optim='NEWTON', learning_rate = 0.5, eps=0.0001,
                              X_test=X_test, Y_test=labels_test)
solver_newton.train(X_train, labels_train)
        

    


            
        
        
        