# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 22:35:46 2021

@author: Rivuozil
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def SpreadingCodeGen(N, m, Nd, dv):
    index = np.zeros((dv, N), dtype=int)
    index_list = []
    for i in range(N):
        index[:, i] = np.random.choice(m, dv, replace=False)
        index[:, i] = np.sort(index[:, i])

        index_list.append(index[:, i].tolist())
        for j in range(i):
            while index_list[j] == index_list[i]:
                index_list[j] = np.sort(np.random.choice(
                    m, dv, replace=False)).tolist()
    return index_list


def CodebookGen(N, m, Nd, dv, SC):
    Codebook = np.zeros([m, m*N*Nd], dtype='int64')
    for i in range(N):
        for j in range(dv):
            for k in range(Nd):
                Codebook[SC[i][j], (SC[i][j] + k*m + m*Nd*i)] = 1
    return Codebook


# return
# @ray.remote
def TrainingGen(N, m, Nd, dv, p, k, snr, Codebook):
    # row: user, col: Nd symbols with m channel gains
    active_index_matrix = np.zeros((k, p), dtype='int32')   # List
    # np array generated by active_index_matrix
    active_delta_matrix = np.zeros((N, p), dtype='int32')
    y_hat_p = np.zeros((2*m*Nd, p))

    # d = 1
    # channel_variance = 10**(-(128.1 + 37.6*np.log10(d))/10)
    channel_variance = 1
    # txPower = 30   # 30dBm = 1Watt
    # txEnergy = 10**(txPower/10) / 1000 #* Ts   # txPower = txEnergy * Fs

    # pldB = 128.1 + 37.6*np.log10(d)
    # rxPower = 30 - pldB
    # rxSNR = snr
    # N0 = rxPower - rxSNR   # dBm
    # N0Linear = 10**(N0/10)
    # channel_variance = 10**(-pldB/10)

    for i in range(p):
        active_index = np.random.choice(N, k, replace=False)
        active_index_matrix[:, i] = active_index
        active_delta_matrix[:, i][active_index] = 1

    x = np.zeros((m*Nd*N, p), dtype='complex64')
    for i in range(p):
        # print(i)
        x_temp = np.zeros((N, m*Nd), dtype='complex64')
        for j in (active_index_matrix[:, i]):
            # bits = np.random.randint(0, 2, size=[1, Nd])*2-1
            bits = np.ones((1, 7))
            channel = np.sqrt(channel_variance/2) * (np.random.randn(1, m) +
                                                      1j*np.random.randn(1, m))
            # channel = np.ones((1, m))
            x_temp[j, :] = np.kron(bits, channel)
        x[:, i] = x_temp.reshape(m*Nd*N,)

    y_tilde = np.dot(Codebook, x)
    rxPwr = np.mean(np.square(np.abs(y_tilde)))
    # noisePwr = 10**(rxPwr) - 10**(snr/10)
    noiseLinear = rxPwr / 10**(snr/10)
    noise = np.sqrt(noiseLinear/2) * (np.random.randn(*y_tilde.shape,) +
                                      1j*np.random.randn(*y_tilde.shape,))
    
    # noise = np.sqrt(noiseLinear/2) * np.zeros(y_tilde.shape,)    # check (no noise)
    # noisePwr = np.mean(np.square(np.abs(noise)))   # check (noise power)

    y_tilde = y_tilde + noise

    y_hat_p[:m, :] = np.real(y_tilde)
    y_hat_p[m:, :] = np.imag(y_tilde)

    y_hat_p = y_hat_p.T
    active_delta_matrix = active_delta_matrix.T
    active_delta_matrix = active_delta_matrix/k
    return y_hat_p, active_delta_matrix


def Hidden_Layer(input_tensor, alpha, stage):
    """
    Parameters
    ----------
    input_tensor : Output of last layer
    alpha : Number of neuron
    stage : Index of hidden layer

    Returns output tensor
    -------
    """
    name_base = 'HL' + stage
    x = layers.Dense(alpha, name=name_base + '_1',
                     # kernel_initializer='ones',
                     # kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                     activity_regularizer=tf.keras.regularizers.l2(0.00005)
                     )(input_tensor)
    # x = layers.Dense(alpha, name=name_base + '_1')(input_tensor)
    x = layers.BatchNormalization(name=name_base + '_2')(x)
    x = layers.Activation('relu', name=name_base + '_3')(x)
    x = layers.Dropout(0.1, name=name_base + '_4')(x)
    x = layers.add([x, input_tensor], name=name_base + '_Add')
    return x


def AUD(alpha, N, m, Nd):
    model_input = layers.Input(shape=[2*m*Nd, ], name='InputLayer')
    x = layers.Dense(alpha, name='InputFC')(model_input)
    x = layers.BatchNormalization(name='InputBN')(x)
    x = Hidden_Layer(x, alpha, stage='_A')
    x = Hidden_Layer(x, alpha, stage='_B')
    x = Hidden_Layer(x, alpha, stage='_C')
    x = Hidden_Layer(x, alpha, stage='_D')
    x = Hidden_Layer(x, alpha, stage='_E')
    x = Hidden_Layer(x, alpha, stage='_F')
    # x = Hidden_Layer(x, alpha, stage='_G')
    # x = Hidden_Layer(x, alpha, stage='_H')
    # x = Hidden_Layer(x, alpha, stage='_I')
    # x = Hidden_Layer(x, alpha, stage='_J')
    # x = Hidden_Layer(x, alpha, stage='_K')
    # x = Hidden_Layer(x, alpha, stage='_L')

    x = layers.Dense(N, name='OutputFC')(x)
    x = layers.Softmax(name='OutputActivatio')(x)

    model = Model(model_input, x, name='D_AUD')
    return model
