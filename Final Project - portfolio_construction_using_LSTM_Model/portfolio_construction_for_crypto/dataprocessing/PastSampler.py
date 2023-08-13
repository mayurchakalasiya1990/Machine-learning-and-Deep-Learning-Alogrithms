# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 19:03:57 2022

@author: ChakalasiyaMayurVash
"""

import numpy as np
 
class PastSampler:
    '''
    Forms training samples for predicting future values from past value
    '''
     
    def __init__(self, N, K, sliding_window = True):
        '''
        Predict K future sample using N previous samples
        '''
        self.K = K
        self.N = N
        self.sliding_window = sliding_window
 
    def transform(self, A):
        M = self.N + self.K     #Number of samples per row (sample + target)
        #indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0]%M == 0:
                I = np.arange(M)+np.arange(0,A.shape[0],M).reshape(-1,1)
                
            else:
                I = np.arange(M)+np.arange(0,A.shape[0] -M,M).reshape(-1,1)
                print(I)
            
        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        print('A[I]:',A[I])
        print('B:',B)
        
        ci = self.N * A.shape[1]    #Number of features per sample
        print('ci',ci)
        r1=B[:, :ci]
        r2=B[:, ci:]
        print('r1',r1)
        print('r2',r2)
        return B[:, :ci], B[:, ci:] #Sample matrix, Target matrix

