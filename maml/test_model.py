import os
import argparse
import numpy as np
import tensorflow as tf

from maml import STDN, AttentionSTDN
from maml_data_generator import DataGenerator
from sklearn.metrics import mean_squared_error


def test(model, data_generator, sess, saver):
    train_inputs, train_labels = data_generator.get_all_data(purpose='train')
    train_inputs, train_labels = train_inputs[0], train_labels[0]

    test_inputs, test_labels = data_generator.get_all_data(purpose='test')
    test_inputs, test_labels = test_inputs[0], test_labels[0]

    train_batch_num = int(train_inputs.shape[1] / update_batch_size)
    test_batch_num = int(test_inputs.shape[1] / update_batch_size)

    if len(output_dir) > 0:
        data_generator.save_test_ground_truth(output_dir=output_dir,
                                              test_data_num=test_batch_num * update_batch_size)
    for epoch in range(epochs):
        total_test_loss = []
        total_outputa = []

        for i in range(test_batch_num):
            inputa = test_inputs[:, i * update_batch_size: (i+1) * update_batch_size, :, :]
            labela = test_labels[:, i * update_batch_size: (i+1) * update_batch_size, :]
            if "att" in model_type:
                dummy_clusters = np.zeros(shape=(len(inputa), update_batch_size, 1))
                feed_dict = {model.inputa: inputa, model.labela: labela, model.clustera: dummy_clusters}
            else:
                feed_dict = {model.inputa: inputa, model.labela: labela}
            outputa, loss1, = sess.run([model.outputas, model.total_loss1], feed_dict)
            total_outputa.append(outputa)
            total_test_loss.append(loss1)
        total_outputa = np.concatenate(total_outputa, axis=1)

        if len(output_dir) > 0:
            np.savez(output_dir + "/output_" + model_type, total_outputa)
            saver.save(sess, output_dir + "/model_" + model_type)
        print(epoch, np.sqrt(np.mean(total_test_loss)))

        total_train_loss = []
        total_train_outputa = []

        for i in range(train_batch_num):
            inputa = train_inputs[:, i * update_batch_size: (i + 1) * update_batch_size, :, :]
            labela = train_labels[:, i * update_batch_size: (i + 1) * update_batch_size, :]
            if "att" in model_type:
                dummy_clusters = np.zeros(shape=(len(inputa), update_batch_size, 1))
                feed_dict = {model.inputa: inputa, model.labela: labela, model.clustera: dummy_clusters}
            else:
                feed_dict = {model.inputa: inputa, model.labela: labela}
            sess.run([model.finetune_op], feed_dict)
            outputa, loss1 = sess.run([model.outputas, model.total_loss1], feed_dict)
            total_train_outputa.append(outputa)
            total_train_loss.append(loss1)

        if len(output_dir) > 0:
            np.savez(output_dir + "/output_train_" + model_type, total_train_outputa)
        print("train:", epoch, np.sqrt(np.mean(total_train_loss)))

def main():
    tf.set_random_seed(1234)

    print(model_type, "att" in model_type, "meta" in model_type)
    if "att" in model_type:
        model = AttentionSTDN(dim_input=dim_input, dim_output=dim_output, seq_length=seq_length,
                              filter_num=64, dim_cnn_flatten=7*7*64,
                              dim_fc=512, dim_lstm_hidden=128,
                              update_lr=update_lr, meta_lr=meta_lr,
                              meta_batch_size=1,
                              update_batch_size=update_batch_size,
                              test_num_updates=1,
                              cluster_num=4, memory_dim=mem_dim,
                              cluster_loss_weight=0)
    else:
        model = STDN(dim_input=dim_input, dim_output=dim_output, seq_length=seq_length,
                     filter_num=64, dim_cnn_flatten=7*7*64,
                     dim_fc=512, dim_lstm_hidden=128,
                     update_lr=update_lr, meta_lr=meta_lr,
                     meta_batch_size=1,
                     update_batch_size=update_batch_size,
                     test_num_updates=1)
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
        data_generator.load_train_data(cities=[city], train_prop=int(test_days*24), select_data='all', shuffle=False)
    else:
        data_generator.load_train_data(cities=[city], train_prop=int(test_days*24), select_data='pick', shuffle=False)

    if len(save_dir) > 0:
        model_file = save_dir + "/" + model_type + "/" + test_model_name
        saver.restore(sess, model_file)

        print("Testing:", model_file, "with %d days data" % test_days)
    else:
        print("Target data only", "with %d days data" % test_days)
    test(model, data_generator, sess, saver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='nyc,dc')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--model_type', type=str, default='')

    parser.add_argument('--test_model', type=str)
    parser.add_argument('--test_days', type=int)
    parser.add_argument('--output_dir', type=str, default='')

    parser.add_argument('--update_batch_size', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--update_lr', type=float, default=1e-5)

    parser.add_argument('--mem_dim', type=int, default=8)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu_id', type=str, default="4")

    dim_output = 2
    dim_input = 7*7*dim_output
    seq_length = 8

    args = parser.parse_args()

    city = args.city
    save_dir = args.save_dir
    model_type = args.model_type

    mem_dim = args.mem_dim

    test_model_name = args.test_model
    test_days = args.test_days
    output_dir = args.output_dir

    update_batch_size = args.update_batch_size
    threshold = args.threshold
    meta_lr = args.meta_lr
    update_lr = args.update_lr

    epochs = args.epochs

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    main()
