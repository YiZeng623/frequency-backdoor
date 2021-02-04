from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam,Adadelta
import numpy as np
import tensorflow as tf
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

import random
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')


def cifar10_model():
    weight_decay = 1e-4
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32,32,3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),name='last_conv'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',name='dense'))

    model.load_weights('./models/cifar10_6_cnn.h5py')
    opt = Adadelta(lr=0.05)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def patching_test(clean_sample, attack_name):
    '''
    this code conducts a patching procedure to generate backdoor data
    **please make sure the input sample's label is different from the target label

    clean_sample: clean input
    attack_name: from attack_list: {'l2_inv', 'l0_inv', 'trojan_wm', 'trojan_sq', 'badnets'}
    '''
    attack_list = {'l2_inv', 'l0_inv', 'trojan_wm', 'trojan_sq', 'badnets', 'nature'}
    assert attack_name in attack_list
    if attack_name == 'badnets':
        output = np.copy(clean_sample)
        pat_size = 4
        output[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 1
    else:
        trimg = plt.imread('./triggers/' + attack_name + '.png')
        if attack_name == 'l0_inv':
            mask = 1 - np.transpose(np.load('./triggers/mask.npy'), (1, 2, 0))
            output = clean_sample * mask + trimg
        else:
            output = clean_sample + trimg
    output[output > 1] = 1
    return output

def dct_test_data(x_test,attackname):
    poi_test = np.zeros_like(x_test)
    attack_list = {str(attackname)}
    for i in range(x_test.shape[0]):
        attack_name = random.sample(attack_list, 1)[0]
        poi_test[i] = patching_test(x_test[i], attack_name)

    x_dct_test = np.vstack((x_test, poi_test))  # [:,:,:,0]
    y_dct_test = (np.vstack((np.zeros((x_test.shape[0], 1)), np.ones((x_test.shape[0], 1))))).astype(np.int)
    for i in range(x_dct_test.shape[0]):
        for channel in range(3):
            x_dct_test[i][:, :, channel] = dct2((x_dct_test[i][:, :, channel] * 255).astype(np.uint8))
    hot_test_lab = np.squeeze(np.eye(2)[y_dct_test])
    return x_dct_test,hot_test_lab