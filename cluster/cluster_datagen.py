import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class Data_Get():

    def __init__(self, normalization=False):
        self.city = ['dc', 'nyc']
        self.citysize = {'dc': [16, 16], 'nyc': [10, 20]}
        self.seqdata = []
        self.train_data_info = []
        self.normalization = normalization
        self.Load_data()

    def Load_data(self):
        for idxcity, eachcity in enumerate(self.city):
            data = np.load(
                open('../maml/raw_data/bike_{}_{}_{}_3600.npz'.format(eachcity, self.citysize[eachcity][0],
                                                               self.citysize[eachcity][1]),
                     "rb"))['traffic'][:, :, :, 0]

            all_time, idx, idy = data.shape

            all_round = int(all_time / 24)

            data = data[:all_round * 24].reshape((all_round, 24, idx, idy))

            data = np.mean(data, axis=0)

            for i in range(idx):
                for j in range(idy):
                    if np.max(data[:, i, j]) > 1:
                        if self.normalization:
                            self.seqdata.append(data[:, i, j]/np.max(data[:,i,j]))
                        else:
                            self.seqdata.append(data[:, i, j])
                        self.train_data_info.append([i, j, idxcity])
        self.seqdata = np.array(self.seqdata)
        return self.seqdata

    def Get_slope(self):
        newdata = np.zeros((self.seqdata.shape))
        n = self.seqdata.shape[1]
        newdata[:, :n - 1] = self.seqdata[:, 1:] - self.seqdata[:, :n - 1]
        newdata = newdata[:, :, np.newaxis]
        olddata = self.seqdata[:, :, np.newaxis]
        newdata = np.concatenate((newdata, olddata), axis=2)

        return newdata
