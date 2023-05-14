__author__ = 'Haohan Wang'

import numpy as np
from BaseModel import BaseModel
from LMM import LinearMixedModel

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error as mse

from tqdm import tqdm
import sys

class PersonalizedThroughMixedModel():
    def __init__(self, mode_regressionModel='lmm', regWeight=0):

        self.mode_regressionModel = mode_regressionModel
        self.regWeight = regWeight

        self.corrector = BaseModel()

        if self.mode_regressionModel == 'lmm':
            self.regressor = LinearMixedModel(fdr=False)
        elif self.mode_regressionModel == 'lr':
            self.regressor = LinearRegression()
        elif self.mode_regressionModel == 'lasso':
            self.regressor = Lasso(alpha = self.regWeight)
        else:
            print("only support lmm, lr, lasso")
            sys.exit()

    def fit(self, X, y, C):

        K = np.zeros([X.shape[0], X.shape[0]])
        D = np.diag(np.dot(X, X.T))

        # print (D)

        B = np.zeros_like(X)
        P = np.zeros_like(X)

        for i in tqdm(range(X.shape[0])):
            sig = np.mean(np.square(C - C[i]), 1) # todo: sum follows the derivation in the notes,
                                                 # todo: but an annoying fact is that this scales with the dimension of C
                                                 # todo: so maybe a mean here now?

            ### save computation when the error terms are the same
            idx = np.where(sig==0)[0][0]

            if idx < i:
                B[i] = B[idx]
                P[i] = P[idx]
            else:
                diag = sig*D

                # print (i)

                np.fill_diagonal(K, diag)

                Xc, Yc = self.corrector.correctData(X, y, K)

                if self.mode_regressionModel == 'lmm':
                    self.regressor.setK(np.dot(Xc, Xc.T))
                    self.regressor.fit(Xc, Yc)
                    beta = self.regressor.getBeta()
                    P[i] = np.exp(-self.regressor.getNegLogP())
                else:
                    self.regressor.fit(Xc, Yc)
                    beta = self.regressor.coef_

                B[i] = beta
        
        self.B = B
        self.P = P
        return B, P
    def predict(self, X, Y):

        return 

