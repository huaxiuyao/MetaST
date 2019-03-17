import numpy as np
import matplotlib.pyplot as plt

city = ['dc', 'nyc']
citysize = {'dc': [16, 16], 'nyc': [10, 20]}

all_trained_data = []

for idx, eachcity in enumerate(city):
    data = np.load(
        open('../maml/raw_data/bike_{}_{}_{}_3600.npz'.format(eachcity, citysize[eachcity][0], citysize[eachcity][1]), "rb"))[
               'traffic'][:, :, :, 0]

    all_time, idx, idy = data.shape

    all_round = int(all_time / 24)

    xaxis = np.arange(24)

    data = data[:all_round * 24].reshape((all_round, 24, idx, idy))

    data = np.mean(data, axis=0)

    for i in range(idx):
        for j in range(idy):
            if np.max(data[:, i, j]) > 1:
                all_trained_data.append([data[:, i, j]] + [i, j, idx])
                plt.plot(xaxis, data[:, i, j])
                plt.show()

all_trained_data = np.array(all_trained_data)
print(all_trained_data.shape)
