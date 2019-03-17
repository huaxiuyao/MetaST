import os
import argparse
import numpy as np
import tensorflow as tf

from maml import STDN, AttentionSTDN
from maml_data_generator import DataGenerator


def train(model, data_generator, sess, saver):
    for epoch in range(iterations):
        if "att" in model_type:
            batch_x, batch_y, batch_c, batch_cw = data_generator.generate(purpose='train', with_cluster=True,
                                                                          update_batch_size=update_batch_size)
            inputa, labela, clustera, cluster_weighta = batch_x, batch_y, batch_c, batch_cw
            batch_x, batch_y, batch_c, batch_cw = data_generator.generate(purpose='test', with_cluster=True,
                                                                          update_batch_size=update_batch_size)
            inputb, labelb, clusterb, cluster_weightb = batch_x, batch_y, batch_c, batch_cw
            feed_dict = {model.inputa: inputa, model.inputb: inputb,
                         model.labela: labela, model.labelb: labelb,
                         model.clustera: clustera, model.clusterb: clusterb,
                         model.cluster_weighta: cluster_weighta, model.cluster_weightb: cluster_weightb}
        else:
            batch_x, batch_y = data_generator.generate(purpose='train', with_cluster=False,
                                                       update_batch_size=update_batch_size)
            inputa, labela = batch_x, batch_y
            batch_x, batch_y = data_generator.generate(purpose='test', with_cluster=False,
                                                       update_batch_size=update_batch_size)
            inputb, labelb = batch_x, batch_y
            feed_dict = {model.inputa: inputa, model.inputb: inputb,
                         model.labela: labela, model.labelb: labelb}

        if epoch % 100 == 0:
            model_file = save_dir + "/" + model_type + "/model_" + str(epoch)
            saver.save(sess, model_file)
            if "att" in model_type:
                res = sess.run([model.total_rmse1, model.total_rmse2,
                                model.total_closs1, model.total_closses2], feed_dict)
            else:
                res = sess.run([model.total_rmse1, model.total_rmse2], feed_dict)
            print(epoch, res)
        else:
            if "meta" in model_type:
                sess.run([model.metatrain_op], feed_dict)
            elif "pretrain" in model_type:
                sess.run([model.pretrain_op], feed_dict)


def main():
    tf.set_random_seed(1234)

    print(model_type, "att" in model_type, "meta" in model_type)
    if "att" in model_type:
        model = AttentionSTDN(dim_input=dim_input, dim_output=dim_output, seq_length=seq_length,
                              filter_num=64, dim_cnn_flatten=7*7*64,
                              dim_fc=512, dim_lstm_hidden=128,
                              update_lr=update_lr, meta_lr=meta_lr,
                              meta_batch_size=len(cities),
                              update_batch_size=update_batch_size,
                              test_num_updates=test_num_updates,
                              cluster_num=4, memory_dim=mem_dim,
                              cluster_loss_weight=cluster_loss_weight)
    else:
        model = STDN(dim_input=dim_input, dim_output=dim_output, seq_length=seq_length,
                     filter_num=64, dim_cnn_flatten=7*7*64,
                     dim_fc=512, dim_lstm_hidden=128,
                     update_lr=update_lr, meta_lr=meta_lr,
                     meta_batch_size=len(cities),
                     update_batch_size=update_batch_size,
                     test_num_updates=test_num_updates)
    model.construct_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    data_generator = DataGenerator(dim_input=dim_input,
                                   dim_output=dim_output,
                                   seq_length=seq_length,
                                   threshold=threshold)
    if dim_output == 2:
        data_generator.load_train_data(cities=cities, train_prop=0.8, select_data='all')
    else:
        data_generator.load_train_data(cities=cities, train_prop=0.8, select_data='pick')

    print("Training:", model_type)
    train(model, data_generator, sess, saver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', type=str, default='nyc,dc')
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--model_type', type=str, default='')

    parser.add_argument('--update_batch_size', type=int, default=128)
    parser.add_argument('--test_num_updates', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--update_lr', type=float, default=1e-5)
    parser.add_argument('--cluster_loss_weight', type=float)
    parser.add_argument('--mem_dim', type=int, default=8)

    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--gpu_id', type=str, default="4")

    dim_output = 2
    dim_input = 7*7*dim_output
    seq_length = 8

    args = parser.parse_args()

    cities = args.cities.split(',')
    save_dir = args.save_dir
    model_type = args.model_type

    update_batch_size = args.update_batch_size
    test_num_updates = args.test_num_updates
    threshold = args.threshold

    cluster_loss_weight = args.cluster_loss_weight
    mem_dim = args.mem_dim

    meta_lr = args.meta_lr
    update_lr = args.update_lr

    iterations = args.iterations

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    main()
