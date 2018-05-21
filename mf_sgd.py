#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  28 01:48:05 2018

@author: J Rishabh Kumar
@content: matrix factorization using gradient descent

"""

import numpy as np
import matplotlib.pyplot as plt

class MF():
    
    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.pred_mat = np.zeros(R.shape)
        self.RMSE_train_after_each_iter = []

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse()
            self.RMSE_train_after_each_iter.append(rmse)
            training_process.append((i, rmse))
            print("Iteration: %d ; RMSE = %.4f" % (i+1, rmse))
        
        return training_process, self.pred_mat

    def rmse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        self.pred_mat = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - self.pred_mat[x, y], 2)
        error /= len(xs)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        if prediction > 5.0:
            prediction = 5.0
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
        

    def plot_RMSE(self):
        plt.plot(range(1,self.iterations + 1),self.RMSE_train_after_each_iter, marker='o', label='Training RMSE')
        #plt.plot(range(1,self.iterations + 1),self.RMSE_test_after_each_iter, marker='v', label='Testing RMSE')
        plt.title('MF with SGD with alpha = %.3f and $\ beta =%.3f' % (self.alpha, self.beta))
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show()
        
    def recTopN(self, user, n):
        """
        recommend top-N items for user u in form of list
        """
        self.n = n
        if user >= 0 & user < 50:
            user_vec = self.pred_mat[user,:]
        recom = dict()
        for i in range(len(user_vec)):
            temp = []
            
            
            
            
            
        
        
        
    
        
        