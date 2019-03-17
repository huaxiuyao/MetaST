import argparse
import numpy as np


def preprocessing(data_path, cluster_file, short_term_lstm_seq_len=7, last_feature_num=12,
                  nbhd_size=1, cnn_nbhd_size=3):

    volume = np.load(open(data_path, "rb"))['traffic']

    if cluster_file is not None:
        clustering_res = np.load(cluster_file)
        cluster_ids = []

    print(volume.shape)
    print(np.max(volume), np.min(volume))

    data = volume / np.max(volume)
    cnn_features = []
    seq_loc = []

    for i in range(short_term_lstm_seq_len):
        cnn_features.append([])
    short_term_lstm_features = []
    labels = []

    time_start = last_feature_num + 10
    time_end = data.shape[0]
    volume_type = data.shape[-1]

    for t in range(time_start, time_end):
        if t % 100 == 0:
            print("Now sampling at {0} timeslots.".format(t))
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                # sample common (short-term) lstm
                short_term_lstm_samples = []
                for seqn in range(short_term_lstm_seq_len):
                    # real_t from (t - short_term_lstm_seq_len) to (t-1)
                    real_t = t - (short_term_lstm_seq_len - seqn)

                    # cnn features, zero_padding
                    cnn_feature = np.zeros((2 * cnn_nbhd_size + 1, 2 * cnn_nbhd_size + 1, volume_type))
                    # actual idx in data
                    for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                        for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                            # boundary check
                            if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                continue
                            # get features
                            cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size),
                            :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                    cnn_features[seqn].append(cnn_feature)

                    # lstm features
                    # nbhd feature, zero_padding
                    nbhd_feature = np.zeros((2 * nbhd_size + 1, 2 * nbhd_size + 1, volume_type))
                    # actual idx in data
                    for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                        for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                            # boundary check
                            if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                continue
                            # get features
                            nbhd_feature[nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size), :] = data[real_t,
                                                                                                  nbhd_x, nbhd_y, :]
                    nbhd_feature = nbhd_feature.flatten()
                    last_feature = data[real_t - last_feature_num: real_t, x, y, :].flatten()
                    # hist feature
                    # hist_feature = data[
                    #                real_t - hist_feature_daynum * self.timeslot_daynum: real_t: self.timeslot_daynum,
                    #                x, y, :].flatten()
                    # feature_vec = np.concatenate((hist_feature, last_feature))
                    feature_vec = np.concatenate((last_feature, nbhd_feature))
                    short_term_lstm_samples.append(feature_vec)
                short_term_lstm_features.append(np.array(short_term_lstm_samples))

                if cluster_file is not None:
                    cluster_ids.append(clustering_res[x, y])
                # label
                seq_loc.append([x, y])
                labels.append(data[t, x, y, :].flatten())

    for i in range(short_term_lstm_seq_len):
        cnn_features[i] = np.array(cnn_features[i])

    short_term_lstm_features = np.array(short_term_lstm_features)
    labels = np.array(labels)
    seq_loc = np.array(seq_loc)

    if cluster_file is not None:
        return cnn_features, cluster_ids, short_term_lstm_features, labels, volume, seq_loc
    else:
        return cnn_features, short_term_lstm_features, labels, volume, seq_loc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./raw_data')
    parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--cluster_file', type=str, default='')
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--cnn_nbhd_size', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='./test_data')
    parser.add_argument('--save_filename', type=str, default='')
    args = parser.parse_args()

    if len(args.cluster_file) > 0:
        cnnx, cids, x, y, volume, seq_loc = preprocessing(
            data_path=args.data_dir + "/" + args.filename,
            cluster_file=args.data_dir + "/" + args.cluster_file,
            short_term_lstm_seq_len=args.seq_len,
            nbhd_size=1, cnn_nbhd_size=args.cnn_nbhd_size)
        np.savez(args.save_dir + "/" + args.save_filename,
                 cnnx=cnnx, cids=cids, x=x, y=y, max_volume=np.max(volume),
                 volume_size=volume.shape, seq_loc=seq_loc)
    else:
        cnnx, x, y, volume, seq_loc = preprocessing(
            data_path=args.data_dir + "/" + args.filename,
            cluster_file=None, short_term_lstm_seq_len=args.seq_len,
            nbhd_size=1, cnn_nbhd_size=args.cnn_nbhd_size)
        np.savez(args.save_dir + "/" + args.save_filename,
                 cnnx=cnnx, x=x, y=y, max_volume=np.max(volume),
                 volume_size=volume.shape, seq_loc=seq_loc)


if __name__ == '__main__':
    main()
