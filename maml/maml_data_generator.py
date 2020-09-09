""" Code for loading data. """
import numpy as np

class DataGenerator(object):
    def __init__(self, dim_input, dim_output, seq_length, threshold):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.seq_length = seq_length
        self.threshold = threshold

        self.cnnx_train = {}
        self.cnnx_test = {}
        self.y_train = {}
        self.y_test = {}
        self.cid_train = {}
        self.cid_test = {}

        self.seq_loc = {}
        self.cities = None
        self.cid_weight = None

    def get_test_seq_loc(self):
        return self.seq_loc[self.cities[0]]

    def load_train_data(self, cities, train_prop=0.8, select_data="all", shuffle=False):
        self.cnnx_train = {}
        self.cnnx_test = {}
        self.y_train = {}
        self.y_test = {}

        self.cities = cities
        for city in cities:
            seq_data = np.load('./test_data/%s_seq.npz' % city)
            if 'cids' in seq_data:
                cids = np.array(seq_data['cids'])
            else:
                cids = None
            data = np.swapaxes(np.array(seq_data['cnnx']), 0, 1)
            labels = np.array(seq_data['y'])

            if 'seq_loc' in seq_data:
                self.seq_loc[city] = seq_data['seq_loc']

            if shuffle:
                np.random.seed(1234)
                perm = np.random.permutation(data.shape[0])
                data = data[perm]
                labels = labels[perm]

            max_volume = float(seq_data['max_volume'])
            norm_threshold = self.threshold / max_volume

            if select_data == "pick":
                data = data[:, :, :, :, 0:1]
                labels = labels[:, 0:1]
                valid_idx = labels[:, 0] >= norm_threshold

            elif select_data == "drop":
                data = data[:, :, :, :, 1:2]
                labels = labels[:, 1:2]
                valid_idx = labels[:, 0] >= norm_threshold
            else:
                valid_idx = np.logical_and(labels[:, 0] >= norm_threshold, labels[:, 1] >= norm_threshold)
            if isinstance(train_prop, float):
                data = data[valid_idx]
                labels = labels[valid_idx]

                split_point = int(data.shape[0] * train_prop)

                if cids is not None:
                    cids = cids[valid_idx]
                    self.cid_train[city] = cids[:split_point]
                    self.cid_test[city] = cids[split_point:]

                self.cnnx_train[city] = data[:split_point]
                self.cnnx_train[city] = np.reshape(self.cnnx_train[city],
                                                   (self.cnnx_train[city].shape[0], self.seq_length, -1))
                self.cnnx_test[city] = data[split_point:]
                self.cnnx_test[city] = np.reshape(self.cnnx_test[city],
                                                  (self.cnnx_test[city].shape[0], self.seq_length, -1))
                self.y_train[city] = labels[:split_point]
                self.y_test[city] = labels[split_point:]

            elif isinstance(train_prop, int):
                volume_size = seq_data['volume_size']
                split_point = train_prop * volume_size[1] * volume_size[2]

                train_valid_idx = valid_idx[:split_point]
                test_valid_idx = valid_idx[split_point:]

                if cids is not None:
                    self.cid_train[city] = cids[:split_point][train_valid_idx]
                    self.cid_test[city] = cids[split_point:][test_valid_idx]

                self.cnnx_train[city] = data[:split_point][train_valid_idx]
                self.cnnx_train[city] = np.reshape(self.cnnx_train[city],
                                                   (self.cnnx_train[city].shape[0], self.seq_length, -1))
                self.cnnx_test[city] = data[split_point:][test_valid_idx]
                self.cnnx_test[city] = np.reshape(self.cnnx_test[city],
                                                  (self.cnnx_test[city].shape[0], self.seq_length, -1))
                self.y_train[city] = labels[:split_point][train_valid_idx]
                self.y_test[city] = labels[split_point:][test_valid_idx]

            print(city, "train data shape:", self.cnnx_train[city].shape)
            print(city, "train label shape:", self.y_train[city].shape)
            print(city, "test data shape:", self.cnnx_test[city].shape)
            print(city, "test label shape:", self.y_test[city].shape)
            if cids is not None:
                print(city, "train cluster shape:", self.cid_train[city].shape)
                print(city, "test cluster shape:", self.cid_test[city].shape)
                self.cid_weight = np.zeros(4)
                for cid in self.cid_train[city]:
                    self.cid_weight[int(cid)] += 1
                for cid in self.cid_test[city]:
                    self.cid_weight[int(cid)] += 1
        if self.cid_weight is not None:
            self.cid_weight = 1.0 / self.cid_weight
            self.cid_weight /= self.cid_weight.sum()
            print("cluster weight:", self.cid_weight)

    def save_test_ground_truth(self, output_dir, test_data_num):
        if "bike" in self.cities[0]:
            np.savez(output_dir + "/output_bike_oracle", self.y_test[self.cities[0]][:test_data_num])
        else:
            np.savez(output_dir + "/output_oracle", self.y_test[self.cities[0]][:test_data_num])

    def generate(self, purpose, with_cluster, update_batch_size):
        inputs = np.zeros([len(self.cities), update_batch_size, self.seq_length, self.dim_input])
        outputs = np.zeros([len(self.cities), update_batch_size, self.dim_output])
        constrains = np.zeros([len(self.cities), update_batch_size])
        constrain_weights = np.zeros([len(self.cities), update_batch_size])

        if purpose == "train":
            cnnx = self.cnnx_train
            y = self.y_train
            cid = self.cid_train
        else:
            cnnx = self.cnnx_test
            y = self.y_test
            cid = self.cid_test
        for i, city in enumerate(self.cities):
            total_data_num = cnnx[city].shape[0]
            idx = np.random.choice(total_data_num, update_batch_size, replace=False)
            seqs = cnnx[city][idx]
            labels = np.array(y[city])[idx]
            inputs[i] = seqs
            outputs[i] = labels
            if with_cluster:
                clusters = np.array(cid[city], dtype=int)[idx]
                cluster_weights = self.cid_weight[clusters]
                constrains[i] = clusters
                constrain_weights[i] = cluster_weights
                # constrain_weights[i] = np.ones(shape=cluster_weights.shape)
        if with_cluster:
            return inputs, outputs, constrains, constrain_weights
        else:
            return inputs, outputs

    def get_all_data(self, purpose):
        if purpose == "train":
            cnnx = self.cnnx_train
            y = self.y_train
        else:
            cnnx = self.cnnx_test
            y = self.y_test

        return ([np.array([cnnx[city]]) for city in self.cities],
                [np.array([y[city]]) for city in self.cities])


if __name__ == '__main__':
    data_generator = DataGenerator(dim_input=7*7*2,
                                   dim_output=2,
                                   seq_length=8,
                                   threshold=0)
    data_generator.load_train_data(cities=['nyc'], train_prop=3 * 24, select_data='all')
    print(data_generator.generate(purpose='train', with_cluster=True, update_batch_size=5))


