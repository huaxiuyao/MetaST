import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class Environment_Data_Get():

    def __init__(self, normalization=False):
        self.city = ['midwest', 'west']
        self.citysize = {'midwest': [35, 25], 'west': [30, 25]}
        self.seqdata = []
        self.train_data_info = []
        self.normalization = normalization
        self.Load_data()

    def Load_data(self):
        for idxcity, eachcity in enumerate(self.city):
            data = np.load(
                open('../maml/environment_data/environment_{}_{}_{}_12'.format(eachcity, self.citysize[eachcity][0],
                                                               self.citysize[eachcity][1]),
                     "rb"))['environment'][:, :, :]

            all_time, idx, idy = data[12*16:12*36].shape

            start=12*16

            datanew=np.zeros((12,idx,idy))

            for i in range(12):
                cnt=np.zeros((idx, idy))
                for j in range(20):
                    scale=np.nonzero(data[12*j+i])[0]
                    scale2=np.nonzero(data[12*j+i])[1]
                    for k in range(scale.shape[0]):
                        cnt[scale[k], scale2[k]]+=1
                    datanew[i]+=data[12*j+i]
                datanew[i]=datanew[i]/cnt

            # all_round = int(all_time / 12)
            #
            # data = data[:all_round * 12].reshape((all_round, 12, idx, idy))
            #
            # data = np.mean(data, axis=0)
            data=datanew

            for i in range(idx):
                for j in range(idy):
                    if np.max(data[:, i, j]) > 1.5:
                        if self.normalization:
                            self.seqdata.append((data[:, i, j]-np.min(data[:,i,j]))/(np.max(data[:,i,j])-np.min(data[:,i,j])))
                        else:
                            self.seqdata.append(data[:, i, j])
                        self.train_data_info.append([i, j, idxcity])
        self.seqdata = np.array(self.seqdata)

    def Get_slope(self):
        newdata = np.zeros((self.seqdata.shape))
        n = self.seqdata.shape[1]
        newdata[:, :n - 1] = self.seqdata[:, 1:] - self.seqdata[:, :n - 1]
        newdata = newdata[:, :, np.newaxis]
        olddata = self.seqdata[:, :, np.newaxis]
        newdata = np.concatenate((newdata, olddata), axis=2)

        return newdata

if __name__=='__main__':
    data = Environment_Data_Get(normalization=True)
    sample_num=data.seqdata.shape[0]
    print(sample_num)
    axis_x=np.arange(12)
    for i in range(sample_num):
        plt.plot(axis_x,data.seqdata[i,:])
        plt.show()
        #print(data.seqdata.shape)
