import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, LSTM, LeakyReLU
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

import argparse
import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import datetime

import util


class TPA_LSTM(tf.keras.Model):
    def __init__(self, output_horizon, embedding_size, obs_len, filter_num, n_layers):
        super(TPA_LSTM, self).__init__()
        self.embeddings = Sequential([
            Dense(units=embedding_size),
            LeakyReLU()
        ])
        self.embedding_size = embedding_size
        self.lstm = Sequential([LSTM(units=obs_len, use_bias=True, return_sequences=True) for _ in tf.range(n_layers)])
        self.obs_len = obs_len
        self.tpa = TemporalPatternAttention(filter_num, obs_len - 1, embedding_size)
        self.linear_final = Sequential([
            Dense(units=embedding_size),
            Dense(units=output_horizon)
        ])

    def call(self, x):
        batch_size, _, _ = x.shape
        x = self.embeddings(x)  # similar to embedding, expand feature dimensions to embedding_size m
        x = tf.transpose(x, perm=[0, 2, 1])
        h_matrix = tf.zeros([batch_size, self.embedding_size, self.obs_len])

        for i in tf.range(self.embedding_size):
            m = tf.reshape(x[:, i, :], shape=[batch_size, 1, -1])
            h_m = self.lstm(m)[:, -1, :]
            for j in tf.range(batch_size):
                # update h_matrix
                h_matrix = tf.tensor_scatter_nd_update(h_matrix, [[j, i]], tf.reshape(h_m[j], shape=[1, -1]))
        h_matrix = LeakyReLU()(h_matrix)
        ht = tf.reshape(h_matrix[:, :, -1], shape=[batch_size, self.embedding_size, 1])
        h_matrix = h_matrix[:, :, :-1]

        # reshape hidden states h_matrix to a shape like an image (n, h, w, c)
        h_matrix = tf.reshape(h_matrix, shape=[-1, self.embedding_size, self.obs_len - 1, 1])
        vt = self.tpa(h_matrix, ht)
        ht_concat = tf.concat([vt, ht], axis=1)
        prediction = self.linear_final(tf.transpose(ht_concat, perm=[0, 2, 1]))
        return prediction


class TemporalPatternAttention(tf.keras.Model):
    def __init__(self, filter_num, kernel_size, embedding_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_num = filter_num
        self.embedding_size = embedding_size
        self.conv = Conv2D(filters=filter_num, kernel_size=[1, kernel_size], padding='valid')
        self.linear_w = Dense(units=filter_num)

    def call(self, h_matrix, ht):
        H = self.conv(h_matrix)
        H = tf.reshape(H, shape=[-1, self.embedding_size, self.filter_num])
        ht_to_score = tf.transpose(ht, perm=[0, 2, 1])
        ht_to_score = self.linear_w(ht_to_score)
        score = tf.matmul(H, tf.transpose(ht_to_score, perm=[0, 2, 1]))
        alpha = tf.math.sigmoid(score)
        # tf.print('attention scores: {}'.format(tf.squeeze(alpha)))
        alpha = tf.repeat(alpha, repeats=self.filter_num, axis=-1)
        vt = tf.reduce_sum(tf.multiply(H, alpha), axis=1)
        return tf.reshape(vt, shape=[-1, self.filter_num, 1])


def train(X, y, args):
    num_ts, _, num_features = X.shape
    model = TPA_LSTM(args.output_horizon, args.embedding_size, args.num_obs_to_train, args.filter_num, args.n_layers)
    optimizer = Adam()
    random.seed(99)
    X_tr, y_tr, X_test, y_test = util.train_test_split(X, y)
    losses = []

    y_scalar = util.MaxScaler()
    # x_scalar = util.MaxScaler()
    # X_train = x_scalar.fit_transform(X_train)
    y_tr = y_scalar.fit_transform(y_tr)

    # train
    for _ in tqdm.tqdm(range(args.num_epochs)):
        for _ in range(args.step_per_epoch):
            X_train, y_train, X_f, y_f = util.batch_generator(
                X_tr, y_tr, args.num_obs_to_train, args.output_horizon, args.batch_size)
            X_train = tf.convert_to_tensor(X_train)
            y_f = tf.convert_to_tensor(y_f)
            with tf.GradientTape() as tape:
                y_predict = model(X_train)
                loss = MSE(y_f, y_predict)
                losses.append(loss.numpy().flatten())
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # test
    output_horizon = args.output_horizon
    obs_len = args.num_obs_to_train
    x_test = X_test[:, -output_horizon - obs_len:-output_horizon, :].reshape((num_ts, -1, num_features))
    xf_test = X_test[:, -output_horizon:, :].reshape((num_ts, -1, num_features))
    Y_test = y_test[:, -output_horizon - obs_len:-output_horizon].reshape((num_ts, -1))
    Yf_test = y_test[:, -output_horizon:].reshape((num_ts, -1))

    x_test = tf.convert_to_tensor(x_test)
    y_pred = model(x_test)
    y_pred = y_pred.numpy()
    if y_scalar is not None:
        y_pred = y_scalar.inverse_transform(y_pred)
    y_pred = y_pred.ravel()

    loss = np.sqrt(np.sum(np.square(Yf_test - y_pred)))
    print('losses: {}'.format(loss))

    if args.show_plot:
        plt.figure(1, figsize=(20, 5))
        plt.plot([k + output_horizon + obs_len - output_horizon for k in range(output_horizon)], y_pred, "r-")
        plt.title('Prediction uncertainty')
        yplot = y_test[-1, -output_horizon - obs_len:]
        plt.plot(range(len(yplot)), yplot, "k-")
        plt.legend(["prediction", "true", "P10-P90 quantile"], loc="upper left")
        ymin, ymax = plt.ylim()
        plt.vlines(output_horizon + obs_len - output_horizon, ymin, ymax, color="blue",
                   linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.show()
    return losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", "-e", type=int, default=1000)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", "-nl", type=int, default=2)
    parser.add_argument("--filter_num", "-fn", type=int, default=32)
    parser.add_argument("--embedding_size", "-es", type=int, default=24)
    parser.add_argument("--output_horizon", "-oh", type=int, default=30, help='prediction horizon')
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=168, help='observation used to prediction')
    # parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument("--show_plot", "-sp", action="store_true", default=True)
    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--max_scaler", "-max", action="store_true", default=True)
    parser.add_argument("--batch_size", "-b", type=int, default=64)

    args = parser.parse_args()

    if args.run_test:
        data_path = util.get_data_path()
        data = pd.read_csv(os.path.join(data_path, "LD_MT200_hour.csv"), parse_dates=["date"])
        data["year"] = data["date"].apply(lambda x: x.year)
        data["day_of_week"] = data["date"].apply(lambda x: x.dayofweek)
        data = data.loc[(data["date"] >= datetime.datetime(2014, 1, 1, 0, 0)) &
                        (data["date"] <= datetime.datetime(2014, 4, 1, 0, 0))]

        # features = ["hour", "day_of_week"]
        hours = data["hour"]
        dows = data["day_of_week"]
        X = np.c_[np.asarray(hours), np.asarray(dows)]
        num_features = X.shape[1]
        num_periods = len(data)
        X = np.asarray(X).reshape((-1, num_periods, num_features))
        y = np.asarray(data["MT_200"]).reshape((-1, num_periods))
        losses = train(X, y, args)
        if args.show_plot:
            plt.plot(range(len(losses)), losses, "k-")
            plt.xlabel("Period")
            plt.ylabel("Loss")
            plt.show()
