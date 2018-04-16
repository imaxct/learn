# coding=utf-8

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn as tfl
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge

import h5py
import cv2
import numpy as np

h5_file = './imgs/dataset.h5'

model_path = './imgs/captcha_cnn.model'

IMG_W = 60
IMG_H = 20
IMG_C = 1
X_DIM = IMG_W * IMG_H * IMG_C

Y_NUM_CLASS = 10
Y_NUM = 4
Y_DIM = Y_NUM_CLASS * Y_NUM


def get_model():
    network = input_data(shape=[None, IMG_H, IMG_W, IMG_C], dtype=tf.float32, name='input')
    
    network = conv_2d(network, 32, 3, padding='same', activation='relu')
    network = max_pool_2d(network, 2, 2, padding='same')

    network = conv_2d(network, 32, 3, padding='same', activation='relu')
    network = max_pool_2d(network, 2, 2, padding='same')
    
    network = conv_2d(network, 32, 3, padding='same', activation='relu')
    network = max_pool_2d(network, 2, 2, padding='same')

    network = fully_connected(network, 1024, activation='relu')

    network = dropout(network, 0.8)

    def custom_loss(y_pred, y_true):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

    network = fully_connected(network, Y_DIM, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
        loss=custom_loss, name='target')
    return network

def train_model(network):

    f = h5py.File(h5_file, 'r')
    total = f['x'].shape[0]

    model = tfl.DNN(
        network, tensorboard_verbose=0, checkpoint_path='./imgs/',
        max_checkpoints=10)
    
    batch_size = 10000
    for i in range(0, total, batch_size):
        print(i, i + batch_size)
        
        x_data = f['x'][i:i+batch_size]
        y_data = f['y'][i:i+batch_size]

        x_train = x_data[0:int(0.9 * x_data.shape[0]),...]
        y_train = y_data[0:int(0.9 * y_data.shape[0]),...]

        print(y_train[0])
        x_test = x_data[int(0.9 * x_data.shape[0]):,...]
        y_test = y_data[int(0.9 * y_data.shape[0]):,...]

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')
        model.fit(
            {'input': x_train}, {'target': y_train}, n_epoch=20,
            validation_set=({'input': x_test}, {'target': y_test}),
            snapshot_step=100, show_metric=True, run_id='captcha'
        )
    model.save(model_path)
    return model



if __name__ == '__main__':
    model = get_model()
    # model =train_model(model)
    model = tfl.DNN(model)
    model.load(model_path, weights_only=True)
    img = cv2.imread('./imgs/test2.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape((-1,))
    img = img / 255.0
    img = img[np.newaxis, :]
    img = img.reshape(img.shape[0], IMG_H, IMG_W, IMG_C)
    y = model.predict(img)
    print(y.shape)
    print(y[0])
    id1 = np.argmax(y[0, 0*Y_NUM_CLASS : 1 * Y_NUM_CLASS])
    id2 = np.argmax(y[0, 1*Y_NUM_CLASS : 2 * Y_NUM_CLASS])
    id3 = np.argmax(y[0, 2*Y_NUM_CLASS : 3 * Y_NUM_CLASS])
    id4 = np.argmax(y[0, 3*Y_NUM_CLASS : 4 * Y_NUM_CLASS])
    print(id1, id2, id3, id4)