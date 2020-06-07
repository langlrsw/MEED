# -*- coding: utf-8 -*-

# from __future__ import absolute_import, division, print_function  

# from nets
import numpy as np
import h5py
import os.path
from sklearn.metrics import roc_auc_score
 

from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU,ELU,LeakyReLU

from keras.models import Model
from keras.layers import Dense, Activation, Input, Reshape
from keras.layers import Conv1D, Flatten, Dropout
from keras.optimizers import SGD, Adam,Adadelta
from keras.layers.normalization import BatchNormalization
from scipy.io import loadmat
 

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy.optimize import minimize_scalar

import pandas as pd

from keras import regularizers
from keras import initializers

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from keras.engine import *


# from main



import numpy as np
import h5py
import os.path
from sklearn.metrics import roc_auc_score
 

from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, InputLayer, Input, merge,concatenate,add,Lambda,multiply
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU,ELU,LeakyReLU

from keras.models import Model
from keras.layers import Dense, Activation, Input, Reshape
from keras.layers import Conv1D, Flatten, Dropout
from keras.optimizers import SGD, Adam,RMSprop
from keras.layers.normalization import BatchNormalization
from scipy.io import loadmat
 

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy.optimize import minimize_scalar

import pandas as pd
 
 

 
from glob import glob
import time as time_simple
import tensorflow as tf




# from ori


from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Embedding, Dense, Dropout, Activation
from keras.datasets import imdb
from keras.engine.topology import Layer 
from keras import backend as K  
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential 

import numpy as np
import tensorflow as tf 
import time 
import numpy as np 
import sys
import os
import urllib2 
import tarfile
import zipfile 
try:
    import cPickle as pickle
except:
    import pickle
import os 
 
from eval_methods import *
from reader import reader_vector_simple_rand

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

 
if K.backend() == "tensorflow":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session = K.tf.Session(config=config)
    K.set_session(session)


# Set parameters:
tf.set_random_seed(10086)
np.random.seed(10086)

max_features = 5000
maxlen = 400
batch_size = 128
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5
k =64# Number of selected words by MEED.
PART_SIZE = 125
###########################################
###############Load data###################
###########################################

 

def proc_x(X):
    X_new = np.zeros((X.shape[0],X.shape[1]+2,X.shape[2]+2,X.shape[3]))
    for i in range(len(X)):
        x = X[i]
        X_new[i][1:-1,1:-1,:]=x
    return X_new
 
def load_data():
    
    data_path = 'data/fashion_mnist_aligned.npz'
    npz_data = np.load(data_path)
    x_train, y_train, x_val, y_val = npz_data['x_train'], npz_data['y_train'], npz_data['x_test'], npz_data[
        'y_test']
    
    idx = np.where(((y_train == 2)+(y_train == 6))>0)[0]
    x_train = x_train[idx]
    y_train = y_train[idx]
    idx = np.where(((y_val ==2)+(y_val == 6))>0)[0]
    x_val = x_val[idx]
    y_val = y_val[idx]
    
    mx = np.mean(x_train,axis=0,keepdims=True)
    
    
    y_train = y_train >= 3
    y_val = y_val>=3
 
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
 
    if x_train.max() > 2:
        x_train /= 255
        x_val /= 255
         
    num_classes = len(np.unique(y_train))

 
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
 
 
    print('x_train.shape,y_train.shape,x_val.shape,y_val.shape,num_classes', x_train.shape,y_train.shape,x_val.shape,y_val.shape,num_classes)

    return x_train, y_train, x_val, y_val,mx

###########################################
###############Original Model##############
###########################################

def create_original_model(input_shape,num_classes):
    """
    Build the original model to be explained. 

    """
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    return model




def generate_original_preds(train = True): 
    """
    Generate the predictions of the original model on training
    and validation datasets. 

    The original model is also trained if train = True. 

    """
    x_train, y_train, x_val, y_val,mx = load_data() 
    input_shape = x_train[0].shape
    num_classes = y_train.shape[1]
    
    model = create_original_model(input_shape,num_classes)

    if train:
        filepath="models/original.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
            verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = callbacks_list, epochs=12, batch_size=128)

    model.load_weights('./models/original.hdf5', 
        by_name=True) 

    pred_train = model.predict(x_train,verbose = 1, batch_size = 1000)
    pred_val = model.predict(x_val,verbose = 1, batch_size = 1000)
     
    np.save('data/pred_train.npy', pred_train)
    np.save('data/pred_val.npy', pred_val) 

###########################################
####################MEED####################
###########################################
# Define various Keras layers.
Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(k), 
    output_shape=lambda x: [x[0],x[2]]) 

Invert = Lambda(lambda x: 1.-x)
Negative = Lambda(lambda x: -x)
def print_shape(x):
    print('shape(x)',x.get_shape())
    return x
PrintShape =  Lambda(lambda x: print_shape(x))

Merge_prior = Lambda(lambda x: (x[2]*x[0]+x[1])/(x[2]+1.), 
    output_shape=lambda x: x[0]) 

# Merge_prior = Lambda(lambda x: x[0], 
#     output_shape=lambda x: x[0]) 

class Concatenate(Layer):
    """
    Layer for concatenation. 
    
    """
    def __init__(self, **kwargs): 
        super(Concatenate, self).__init__(**kwargs)

    def call(self, inputs):
        input1, input2 = inputs  
        input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim] 
        dim1 = int(input2.get_shape()[1])
        input1 = tf.tile(input1, [1, dim1, 1])
        return tf.concat([input1, input2], axis = -1)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        input_shape = list(input_shape2)
        input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
        input_shape[-2] = int(input_shape[-2])
        return tuple(input_shape)
    
def ada_batch_size(mat):
    col = K.tf.reduce_sum(mat, 1)
    col = K.tf.reduce_sum(mat, 1)
    col = K.tf.ones_like(col)
    return K.sum(col)

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
 
def Conv2dT_BN(x, filters, kernel_size, strides=(2,2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

from scipy.special import comb
import itertools


def int2bin(a,bin_size):
    return [int(x) for x in bin(int(a))[2:].zfill(bin_size)]
def gen_vectors_max_inter_angle(d,num_1,M=None,flag_random=False):

    if flag_random:
        if M is None:
            M = 256
        Theta = np.random.randn(M,d)
        Theta /= np.sqrt(np.sum(Theta*Theta,axis=1,keepdims=True))
        return Theta
           
    if M is not None:
        while comb(d,num_1)*2**num_1 < M:
            if num_1>=d:
                break
            num_1 += 1
            

    if d==2 or num_1>=d:
        if M is None:
            M = 256
        
        dangle = 2*np.pi/float(M)
        angle = np.linspace(0,2*np.pi-dangle,M)
        cos = np.cos(angle)[:,np.newaxis]
        sin = np.sin(angle)[:,np.newaxis]
        if d == 2:
            Theta = np.hstack([cos,sin])
        else:
            Theta = np.hstack([cos,sin,np.zeros((M,d-2))])
    else:
        if M is not None and M<=2*d:
            Theta = np.vstack([np.eye(d),-np.eye(d)])
        else:
        
            if num_1 == 1:
                Theta = np.vstack([np.eye(d),-np.eye(d)])
            else:    
                all_combos = list(itertools.combinations(np.arange(d), num_1))

                basic_block = []
                for a in range(2**num_1):
                    x = np.array(int2bin(a,num_1))
                    x = 2*x-1
                    basic_block.append(x)
                basic_block = np.array(basic_block)

                for i_comb,e_comb in enumerate(all_combos):
                    x = np.zeros((2**num_1,d))
                    x[:,list(e_comb)]=basic_block
                    if i_comb == 0:
                        Theta = x
                    else:
                        Theta = np.vstack([Theta,x])
            #     Theta = np.array(Theta)
                Theta /= np.sqrt(np.sum(Theta*Theta,axis=1,keepdims=True))

#     print Theta[:20,:]

    if not flag_random:
#         U = ortho_group.rvs(d)
        a = np.random.randn(d, d)
        U, _ = np.linalg.qr(a)
        Theta = np.dot(Theta,U)
    
    if M is not None and M<Theta.shape[0]:
        idx = np.arange(Theta.shape[0])
        idx = np.random.choice(idx, size=M, replace=False)
        Theta = Theta[idx,:]

    return Theta

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
    return

def my_get_shape(x):
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    return shape_flatten


from scipy.special import logsumexp

def softmax(x,axis=1):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def calculate_acc(pred, y):
    return np.mean(np.argmax(pred, axis = 1) == np.argmax(y, axis = 1))






class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables. 

    """
    def __init__(self, tau0, k, **kwargs): 
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):   
 
        batch_size = K.shape(logits)[0]
        logits_ = K.permute_dimensions(logits, (0,3,1,2))# [batch_size, 1, h, w]
        
        h,w = int(logits_.get_shape()[2]),int(logits_.get_shape()[3])
        unif_shape = [batch_size,self.k,h,w]
 
        
        uniform=K.random_uniform(shape=unif_shape,minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)
 
        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        noisy_logits = tf.reshape(noisy_logits,[-1,self.k,h*w])
        samples = K.softmax(noisy_logits,axis=2)
        noisy_logits = tf.reshape(noisy_logits,[-1,self.k,h,w])
        samples = K.max(noisy_logits, axis = 1) 
        
        logits = tf.reshape(logits,[-1, h, w]) 
 
        output = K.in_train_phase(samples, logits) 
        return tf.expand_dims(output,-1)

    def compute_output_shape(self, input_shape):
        return input_shape

def construct_gumbel_selector(X_ph, h,w,y=None):

    x = X_ph
    if y is not None:
        hy = Dense(h*w, activation='linear')(y)
        hy = Reshape((h,w,1))(hy)
        x = concatenate([x, hy], axis=3)
    
    x=Conv2D(32, kernel_size=(3, 3),
                     activation='relu',padding='same')(x)
    x=Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    logits_T=Conv2D(1, (3, 3), activation='linear',padding='same')(x)
 
    return logits_T

def get_explainer(input_shape,num_classes,k,flag_with_y):
    X_ph = Input(shape=input_shape)
    y = Input(shape=(num_classes,))
    logits_T_prior = Input(shape=(input_shape[0],input_shape[1],1))
    step = Input(shape=(1,1,1))
 
    if flag_with_y:
        logits_T_self = construct_gumbel_selector(X_ph, input_shape[0],input_shape[1],y=y)
    else:
        logits_T_self = construct_gumbel_selector(X_ph, input_shape[0],input_shape[1])
        
        
    logits_T = Merge_prior([logits_T_self,logits_T_prior,step])
    logits_T_output = concatenate([logits_T_self,logits_T],axis=3)
 
    tau = 0.5 
    T = Sample_Concrete(tau, k)(logits_T)
    T_neg = Sample_Concrete(tau, k)(Negative(logits_T))
    
    if flag_with_y:
        E = Model([X_ph,y,logits_T_prior,step], T)
    else:
        E = Model([X_ph,logits_T_prior,step], T)
 
    opt = Adam()  
    E.compile(loss='mse', optimizer=opt)   
    
    if flag_with_y:
        E_neg = Model([X_ph,y,logits_T_prior,step], T_neg)
    else:
        E_neg = Model([X_ph,logits_T_prior,step], T_neg)
 
    opt = Adam()  
    E_neg.compile(loss='mse', optimizer=opt) 
    
    if flag_with_y:
        E_soft = Model([X_ph,y,logits_T_prior,step], logits_T_output)
    else:
        E_soft = Model([X_ph,logits_T_prior,step], logits_T_output)
    opt = Adam()  
    E_soft.compile(loss='mse', optimizer=opt)  
    
    if flag_with_y:
        E_pred = Model([X_ph,y], logits_T_self)
    else:
        E_pred = Model(X_ph, logits_T_self)
    opt = Adam()  
    E_pred.compile(loss='mse', optimizer=opt)  
     
    return E, T, E_neg,T_neg,E_soft,E_pred

def get_approximator_eval(input_shape,num_classes):
    
    X_ph = Input(input_shape)
    x=Conv2D(32, kernel_size=(3, 3),
                     activation='relu')(X_ph)
    x=Conv2D(64, (3, 3), activation='relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Flatten()(x)
    x=Dense(128, activation='relu')(x)
    x=Dropout(0.5)(x)
    preds=Dense(num_classes, activation='softmax')(x)
 
    
    A = Model(X_ph, preds)
    opt = Adadelta()  
    A.compile(loss='categorical_crossentropy', optimizer=opt)   
    
    
    return A 

def get_approximator(input_shape,num_classes):
    
    X_ph = Input(input_shape)
    T = Input(shape=(input_shape[0],input_shape[1],1))
    mean_x = Input(input_shape)
     
    x = subtract([X_ph,mean_x])
    x = multiply([x, T])
    x = add([x,mean_x])
    
    x=Conv2D(32, kernel_size=(3, 3),
                     activation='relu')(x)
    x=Conv2D(64, (3, 3), activation='relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Flatten()(x)
    x=Dense(128, activation='relu')(x)
    x=Dropout(0.5)(x)
    preds=Dense(num_classes, activation='softmax')(x)
 
    
    A = Model(inputs=[X_ph,T,mean_x], outputs=preds)
    opt = Adadelta()  
    A.compile(loss='mse', optimizer=opt)   
    
    
    return A,preds
 

def get_model_pos(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,flag_with_y,mean_x,E_soft,logits_T_prior,step):

    set_trainability(E, False) 
    set_trainability(As, True) 
    set_trainability(Au, True) 

    if flag_with_y:
        T = E([X_ph,y,logits_T_prior,step])
    else:
        T = E([X_ph,logits_T_prior,step])
        
        
    T_neg = Invert(T) 
    preds_s = As([X_ph,T,mean_x])
    preds_u = Au([X_ph,T_neg,mean_x])
    
    output_set = [preds_s,preds_u]
    if flag_with_y:
        model = Model(inputs=[X_ph,y,mean_x,logits_T_prior,step],outputs=output_set)
    else:
        model = Model(inputs=[X_ph,mean_x,logits_T_prior,step],outputs=output_set)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    
    return model,output_set
 

def get_model_neg(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,flag_with_y,mean_x,E_soft,logits_T_prior,step):

    set_trainability(E, True) 
    set_trainability(As, False) 
    set_trainability(Au, False) 

    if flag_with_y:
        T = E([X_ph,y,logits_T_prior,step])
    else:
        T = E([X_ph,logits_T_prior,step])
    T_neg = Invert(T) 
    preds_s = As([X_ph,T,mean_x])
    preds_u = Au([X_ph,T_neg,mean_x])
    
    if flag_with_y:
        logit_T_output = E_soft([X_ph,y,logits_T_prior,step])
    else:
        logit_T_output = E_soft([X_ph,logits_T_prior,step])
    
    output_set = [preds_s,preds_u,logit_T_output]
    
 
    if flag_with_y:
        model = Model(inputs=[X_ph,y,mean_x,logits_T_prior,step],outputs=output_set)
    else:
        model = Model(inputs=[X_ph,mean_x,logits_T_prior,step],outputs=output_set)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    
    return model,output_set

def neg_categorical_crossentropy(y_true, y_pred):
    return -K.mean(K.categorical_crossentropy(y_true, y_pred))


def loss_prior(y_true, y_pred):
    m = y_true
    
    log_z_self,log_z_final = y_pred[:,:,:,:1],y_pred[:,:,:,1:]
    tmp_shape = K.shape(log_z_final)
    d = my_get_shape(log_z_final)
    log_z_final = K.reshape(log_z_final,[tmp_shape[0],d])
    z_final = K.softmax(log_z_final,axis=1)
 
    z_final = K.stop_gradient(z_final)
    log_z_self = K.reshape(log_z_self,[tmp_shape[0],d])
    loss = -K.sum(z_final*log_z_self,axis=1)/(m+1.)
    
    return K.mean(loss)

def sliced_wasserstein_distance_with_Theta(y_true, y_pred,Theta):
     
    y_true_proj = K.dot(y_true,K.transpose(Theta))
    y_pred_proj = K.dot(y_pred,K.transpose(Theta))
    y_true_proj = K.tf.contrib.framework.sort(y_true_proj,axis=0)
    y_pred_proj = K.tf.contrib.framework.sort(y_pred_proj,axis=0)
    loss = K.mean(K.abs(y_pred_proj-y_true_proj))
    
    return loss

flag_anti_swd = True

def SWD_loss_pos(y_true, y_pred):
    c = my_get_shape(y_pred)
     
    Theta_pred = y_true[:,c:]
    y_true = y_true[:,:c]
    
     
    if flag_anti_swd:
        loss_diff_pred = sliced_wasserstein_distance_with_Theta(y_true,y_pred,Theta_pred)
    else:
        loss_diff_pred = K.mean(K.abs(y_true-y_pred))
    
    return K.mean(loss_diff_pred)

def SWD_loss_neg(y_true, y_pred):
    c = my_get_shape(y_pred)
     
    Theta_pred = y_true[:,c:]
    y_true = y_true[:,:c]
    
     
    if flag_anti_swd:
        loss_diff_pred = sliced_wasserstein_distance_with_Theta(y_true,y_pred,Theta_pred)
    else:
        loss_diff_pred = K.mean(K.abs(y_true-y_pred))
    
    return K.mean(-loss_diff_pred)

 
def build_model(input_shape,num_classes,k,flag_with_y):
    X_ph = Input(shape=input_shape)
    mean_x = Input(shape=input_shape)
    y = Input(shape=(num_classes,))
    logits_T_prior = Input(shape=(input_shape[0],input_shape[1],1))
    step = Input(shape=(1,1,1))
    E,_,E_neg,_,E_soft,E_pred=get_explainer(input_shape,num_classes,k,flag_with_y)   
    As,_=get_approximator(input_shape,num_classes)
    Au,_=get_approximator(input_shape,num_classes)
    opt = Adadelta()
    loss_weights = [1.,1e-3]
    loss = ['categorical_crossentropy','categorical_crossentropy']
    model_pos,_ = get_model_pos(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,flag_with_y,mean_x,E_soft,logits_T_prior,step)
    loss_weights = [1.,1e-3,0.]
    loss = ['categorical_crossentropy','categorical_crossentropy',loss_prior]
    model_neg,_ = get_model_neg(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,flag_with_y,mean_x,E_soft,logits_T_prior,step)
    return model_pos,model_neg,E,As,Au,E_pred 

def gradient_explain(M, x, y):
    out = M(x)
    loss = tf.reduce_sum(out*y)
    grad = tf.gradients(loss, x)
    grad_sum = tf.abs(grad[0]*x)
    return grad_sum


def MEED(train = True): 
    
    flag_train_app = True
    flag_with_y = True

    print('Loading dataset...') 
    x_train, y_train, x_val, y_val,mx  = load_data()
    mx = mx*0.
    
    input_shape = x_train[0].shape
    num_classes = y_train.shape[1]
    
    with session as sess:
    
        M = create_original_model(input_shape,num_classes)
        weights_name = [i for i in os.listdir('./models') if i.startswith('original')][0]
        M.load_weights('./models/' + weights_name, 
            by_name=True)  

        pred_train = M.predict(x_train,batch_size=256)
        pred_val = M.predict(x_val,batch_size=256)
 

        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, input_shape[0],input_shape[1],input_shape[2]])
        y_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
        
        # 解释模型
        # 梯度求和  
        grad_sum = gradient_explain(M, x_placeholder, y_placeholder)
        
 
        print('Creating model...')
        model_pos,model_neg,_,As,Au,E = build_model(input_shape,num_classes,k,flag_with_y)

        train_acc = np.mean(np.argmax(pred_train, axis = 1)==np.argmax(y_train, axis = 1))
        val_acc = np.mean(np.argmax(pred_val, axis = 1)==np.argmax(y_val, axis = 1))
        print('The train and validation accuracy of the original model is {} and {}'.format(train_acc, val_acc))

        if train:
 
            epochs = 12
            batch_size = 128
            data_reader_train = reader_vector_simple_rand.Reader(x_train, pred_train, batch_size=batch_size,flag_shuffle=True,rng_seed=123)  


            n_step = int(x_train.shape[0] * epochs / batch_size )
            step_list_sub = np.array([1,2,5]).astype(int) * 100
            step_list = []
            for i_ratio in range(20):
                step_list.extend(step_list_sub)
                step_list_sub = step_list_sub * 10
            step_list.append(n_step-1)

            i_step = -1
            while i_step < n_step:
                i_step += 1
                
                if True:

                    x_batch, y_batch  = data_reader_train.iterate_batch()
                    if x_batch.shape[0]!=batch_size:
                        continue

                    if i_step % 2 == 0:
                        flag_switch = True
                    else:
                        flag_switch = False
                        
                    y_batch_one_hot = to_categorical(np.argmax(y_batch,axis=1),num_classes)
                    
                    selection_prior = sess.run(grad_sum, feed_dict={x_placeholder: x_batch, y_placeholder: y_batch_one_hot})
                    selection_prior = np.reshape(selection_prior,[x_batch.shape[0],-1])
                    selection_prior = softmax(-selection_prior)
                    selection_prior = np.log(selection_prior+1e-40)
                    selection_prior = np.reshape(selection_prior,[x_batch.shape[0],x_batch.shape[1],x_batch.shape[2],1])
                    epoch_cur = 1.#i_step#i_step*x_batch.shape[0]/float(x_train.shape[0])#float(1e8)#float(1e8)#
                    epoch_cur = epoch_cur * np.ones((x_batch.shape[0],1,1,1))

        
                    flag_random = False
                    batch_size_cur = x_batch.shape[0]
                
                    mean_x = np.repeat(mx,x_batch.shape[0],axis=0)
 
                    zzz = y_batch
            

                    if flag_with_y:
                        model_pos.train_on_batch([x_batch,y_batch,mean_x,selection_prior,epoch_cur],[y_batch,zzz])
                    else:
                        model_pos.train_on_batch([x_batch,mean_x,selection_prior,epoch_cur],[y_batch,zzz])
 
 
                    zzz = 1.-y_batch
         

                    if flag_with_y:
                        model_neg.train_on_batch([x_batch,y_batch,mean_x,selection_prior,epoch_cur],[y_batch,zzz,epoch_cur])
                    else:
                        model_neg.train_on_batch([x_batch,mean_x,selection_prior,epoch_cur],[y_batch,zzz,epoch_cur])

    #             if i_step == step_list[-1] or i_step == 1:
                if i_step in step_list:
                    print('------------------------ test ----------------------------')

                    st = time.time()
                    pred_val = M.predict(x_val,batch_size = 2048)
                    duration = time.time() - st
                    print('TPS = {}'.format(duration/x_val.shape[0]))    
                     
                    fidelity,infidelity = eval_without_approximator(E, M, 
                                                                    x_val,pred_val,flag_with_y=flag_with_y,k=k,mx=mx)
                    print('step: %d\tFS-M=%.4f\tFU-M=%.4f'%(i_step,fidelity,infidelity))


                    if flag_train_app:
                        model_As = get_approximator_eval(input_shape,num_classes)
                        model_Au = get_approximator_eval(input_shape,num_classes)
                        fidelity,infidelity,_,_ = eval_with_approximator(E, model_As,model_Au, 
                                                                     x_train, pred_train, x_val, pred_val, epochs=12,
                                                                     flag_with_y=flag_with_y,k=k)
                        print('step: %d\tFS-A=%.4f\tFU-A=%.4f'%(i_step,fidelity,infidelity))

                    sen_N = 50
                    epsilon = 0.2
                    if flag_with_y == False:
                        pred_val = None
                    sensitivity = eval_sensitivity2(x_val, E, epsilon, sen_N, y=pred_val,selection=None)
                    print('step: %d, SEN=%g'%(i_step,sensitivity))
 
    
    return  

if __name__ == '__main__':
  

    import os,signal,traceback
    try:
        MEED()
    except:
        traceback.print_exc()
    finally:
        os.kill(os.getpid(),signal.SIGKILL)

    





