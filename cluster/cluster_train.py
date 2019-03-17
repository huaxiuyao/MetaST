import os

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.contrib.layers import fully_connected
from cluster_datagen_environment import Environment_Data_Get
from CLSTM_AE import T_LSTM_AE
from cluster_datagen import Data_Get
import matplotlib.pyplot as plt
from math import *
import random

learning_rate = 1e-3
iterations = 100
input_dim = 2
fc_output_dim = 16
filter_size = 64
conv_out_dim = 64
hidden_dim = 64
hidden_dim2 = 64
hidden_dim3 = 64
output_dim = hidden_dim
output_dim2 = hidden_dim2
output_dim3 = 64
batch_size = 32

def clustering(data):
    sample_num, seq_len, input_dim = data.shape

    batch_num = int(sample_num / batch_size)

    inputs = tf.placeholder('float', shape=[None, None, input_dim])

    fcoutput = fully_connected(inputs=inputs, num_outputs=fc_output_dim)

    convout = tf.layers.conv1d(inputs=fcoutput, filters=filter_size, kernel_size=3, strides=1, padding='same',
                               data_format="channels_last", activation=tf.tanh)
    lstm_ae = T_LSTM_AE(convout, conv_out_dim, output_dim, output_dim2, output_dim3, hidden_dim, hidden_dim2,
                        hidden_dim3)

    loss_ae = lstm_ae.get_reconstruction_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_ae)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            loss_all = 0
            for j in range(batch_num+1):
                if j<batch_num:
                    feeddata = data[j * batch_size:(j + 1) * batch_size, :]
                else:
                    feeddata = data[j * batch_size:, :]
                _, loss = sess.run([optimizer, loss_ae], feed_dict={inputs: feeddata})
                loss_all += loss
            print('iter {0}, Loss: {1}'.format(i, loss_all))
        data_reps = []
        for j in range(batch_num+1):
            if j < batch_num:
                feeddata = data[j * batch_size:(j + 1) * batch_size, :]
            else:
                feeddata = data[j * batch_size:, :]
            reps, cells = sess.run(lstm_ae.get_representation(), feed_dict={inputs: feeddata})
            if j == 0:
                data_reps = reps
            else:
                data_reps = np.concatenate((data_reps, reps))
    np.save(open('data_reps_norm.npz', 'wb'), data_reps)
    return data_reps


def DTWDistance(s1, s2, w):
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])


def LB_Keogh(s1, s2, r):
    LB_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound) ** 2

    return sqrt(LB_sum)

def k_means_clust(data, num_clust, num_iter, w=3):
    centroids = np.random.choice(data.shape[0], num_clust)
    centroids = data[centroids]
    counter = 0
    for n in range(num_iter):
        print(n)
        counter += 1
        assignments = {}
        # assign data points to clusters
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                if LB_Keogh(i, j, 3) < min_dist:
                    cur_dist = DTWDistance(i, j, w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []

        # recalculate centroids of clusters
        for key in assignments:
            clust_sum = np.zeros(12)
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            #print(len(assignments[0]), len(assignments[1]), len(assignments[2]))
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]

    return centroids, assignments

def main():
    datagen = Environment_Data_Get(normalization=True)
    print(datagen.seqdata.shape)

    # if os.path.exists('data_reps_norm.npz'):
    #     data_reps = np.load(open('data_reps_norm.npz','rb'))
    # else:
    #     data_reps = clustering(data)
    centroids, assignments = k_means_clust(datagen.seqdata, 3, 20)
    kmeans_res = np.zeros(datagen.seqdata.shape[0])
    for eachkey in assignments:
        print(len(assignments[eachkey]))
        for each in assignments[eachkey]:
            kmeans_res[each]=eachkey
    # kmeans_res = KMeans(n_clusters=3, random_state=0, init='k-means++').fit_predict(datagen.seqdata)
    # centroid_values = kmeans.cluster_centers_
    kmeans_res = kmeans_res + 1

    print(kmeans_res.shape)

    x_axis=np.arange(12)

    for idx, eachcity in enumerate(datagen.city):
        print(eachcity)
        res=np.zeros(tuple(datagen.citysize[eachcity]))
        for idxinfo, eachinfo in enumerate(datagen.train_data_info):
            if eachinfo[2]==idx:
                res[eachinfo[0], eachinfo[1]]=kmeans_res[idxinfo]
                # plt.plot(x_axis, datagen.seqdata[idxinfo], 'o-')
                # plt.xlabel('group {}'.format(kmeans_res[idxinfo]))
                # plt.show()
        np.save(open('environment_cluster_res_norm_simple_{}'.format(eachcity),'wb'), res)

if __name__ == '__main__':
    main()
