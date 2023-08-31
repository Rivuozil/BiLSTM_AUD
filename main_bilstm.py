# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 22:37:58 2021

@author: Leo
"""

from generator import SpreadingCodeGen, CodebookGen, TrainingGen, AUD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling1D,Bidirectional,LSTM

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import ray



 

# %%
# Params
m = 70          # num of subcarriers
N = 100         # num of total users
Nd = 7          # num of "multiple measurement"
alpha = 9*N    # num of neurons each dense layer
snr = 10        # training SNR

k = 4           # num of active user
dv = 7         # num of spreading code for each user



# %%
# Training data generation
SC = SpreadingCodeGen(N, m, Nd, dv)
Codebook = CodebookGen(N, m, Nd, dv, SC)

# number of training data: n*nn*p, p is depend on your RAM size, the larger the faster
# when using mega work, you need to specify variable first
# ray.init(num_cpus=20)
n = 20        # workers
nn = 12       # result id (num of ObjectRef)
p = 300

@ray.remote
def Mega_TrainingGen(start, end):
    return [TrainingGen(N, m, Nd, dv, p, k, snr, Codebook) for x in range(start, end)]

# start = time.time()
# a = ([Mega_TrainingGen.remote(0, n) for i in range(nn)])
# a = ray.get(a)
# y_hat_p = np.zeros((1, 2*m))
# active_delta_matrix = np.zeros((1, N))
# for i in range(nn):
#     for j in range(n):
#         y_hat_p = np.vstack((y_hat_p, a[i][j][0]))
#         active_delta_matrix = np.vstack((active_delta_matrix, a[i][j][1]))
# print('stack:', time.time() - start)


# start = time.time()
# y_hat_p = np.zeros((n*nn*p, 2*m))
# active_delta_matrix = np.zeros((n*nn*p, N))
# a = ([Mega_TrainingGen.remote(0, n) for i in range(nn)])
# a = ray.get(a)
# for i in range(nn):
#     for j in range(n):
#         y_hat_p[(i*n+j)*p:(i*n+j+1)*p,:] = a[i][j][0]
#         active_delta_matrix[(i*n+j)*p:(i*n+j+1)*p,:] = a[i][j][1]
# print('index:', time.time() - start)


start = time.time()
result_ids = [Mega_TrainingGen.remote(0, n) for x in range(nn)]
y_hat_p = np.zeros((n*nn*p, 2*m))
active_delta_matrix = np.zeros((n*nn*p, N))
while len(result_ids):
    i = nn - len(result_ids)
    print('i:', i)
    done_id, result_ids = ray.wait(result_ids)
    temp = ray.get(done_id[0])
    print('doneID:', done_id)
    for j in range(n):
        y_hat_p[(i*n+j)*p:(i*n+j+1)*p, :] = temp[j][0]
        active_delta_matrix[(i*n+j)*p:(i*n+j+1)*p, :] = temp[j][1]

print('wait', time.time() - start, 'seconds')
print("shape of yhatp {}".format(y_hat_p.shape))
y_hat_p=y_hat_p.reshape(y_hat_p.shape[0],y_hat_p.shape[1],1)
#y_hat_p=y_hat_p.squeeze()

# %%
# Training
early_stopping = EarlyStopping(monitor='loss', patience=25)
saveWeight = ModelCheckpoint(filepath='./' + 'AUD_' + str(k) + 'user_' + str(dv) + 'dv_' + str(snr) + 'snr_' + '.h5',
                              monitor='loss',
                              # verbose=1,
                              save_best_only=True,
                              save_weights_only=True,
                              mode='auto', save_freq=150)
reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.5, patience=6,
                              min_lr=0.5*10**-7)

adam = Adam(learning_rate=5*10**-4)
M=2*m
model = Sequential()
model.add(Bidirectional(LSTM(alpha,return_sequences=True),input_shape=(M,1)))
model.add(GlobalAveragePooling1D())
model.add(Dense(N,activation='softmax'))

model.build(input_shape=(M,1))

model.summary()

model.compile(optimizer=adam,
             loss='categorical_crossentropy',
             metrics=['accuracy'])
    
model_hist=model.fit(y_hat_p[:2400000:,:], active_delta_matrix[:2400000:,:],
              epochs=30,
              batch_size=2048,
              validation_split=0.25,
              callbacks=[early_stopping, reduce_lr, saveWeight])
model_plot = tf.keras.utils.plot_model(model)

hist_dict = model_hist.history
all_val_loss = hist_dict['val_loss']
all_loss = hist_dict['loss']
all_val_acc = hist_dict['val_accuracy']
all_acc = hist_dict['accuracy']

epoch = np.arange(1, len(all_loss) + 1)
plt.semilogy(epoch, all_val_loss, label='val_loss')
plt.semilogy(epoch, all_loss, label='loss')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('Binary cross-entropy loss')
plt.show()
plt.savefig('Binary_loss_bilstm')

plt.plot(epoch, all_val_acc, label='val_accuracy')
plt.plot(epoch, all_acc, label='accuracy')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
plt.savefig('accuracy_bilstm')

# %%
# Predict Psucc
test = 2500
test_SNR = np.arange(0, 21, 2)
p_succ = np.zeros(test_SNR.shape,)
p_hat_list = []
for i in range(len(test_SNR)):
    print(test_SNR[i])
    test_input, p_real = TrainingGen(
        N, m, Nd, dv, test, k, test_SNR[i], Codebook)
    print("shape of test input {}".format(test_input.shape))
    test_input=np.array(test_input)
    test_input=test_input.reshape(test_input.shape[0],test_input.shape[1],1)
    p_hat = model.predict(test_input)
    p_hat = np.argsort(-p_hat)[:, :k]
    p_hat_list.append(p_hat.tolist())
    p_temp = np.zeros([test, N], dtype='int')
    for j in range(k):
        p_temp[np.arange(test), p_hat[:, j]] = 1
    z = np.where((p_temp.reshape(test*N,)-p_real.reshape(test*N,)) == 0)
    p_succ[i] = np.mean((np.where(p_real != 0)[1] ==
                        np.where(p_temp != 0)[1]).astype(int))
    # print(np.where(
    #     np.where(p_real != 0)[1] != np.where(p_temp != 0)[1])[0].shape)

plt.title('Psucc (dv=' + str(dv) + ', ' + ' k=' + str(k) + ')')
plt.grid('true')
plt.xlabel('SNR')
plt.ylabel('Psucc')
plt.xlim(0, 20)
plt.ylim(0, 1)
plt.xticks(test_SNR, test_SNR)
plt.plot(test_SNR, p_succ, marker='o')
plt.show()
plt.savefig('Psucc_bilstm')
