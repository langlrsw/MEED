# -*- coding: utf-8 -*-

# from __future__ import absolute_import, division, print_function  
 
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
from keras.optimizers import SGD, Adam
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
 
from keras.models import load_model
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

 
from reader import reader_vector_simple_rand
 

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
# from utils import create_dataset_from_score, calculate_acc
 
from eval_methods import *

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
batch_size = 40
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5
k =10 # Number of selected words by MEED.
PART_SIZE = 125
###########################################
###############Load data###################
###########################################

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
    return

def my_get_shape(x):
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    return shape_flatten

def select_word(xs, num_words):
    xs_unfold = np.reshape(xs, -1)
    x_reuters = np.array([2 if idx >= num_words else idx for idx in xs_unfold])
    x_reuters = np.reshape(x_reuters, xs.shape)
    return x_reuters
 
def load_data():
    """
    Load data if data have been created.
    Create data otherwise.

    """

    if 'data' not in os.listdir('.'):
        os.mkdir('data') 
        
    if 'id_to_word.pkl' not in os.listdir('data'):
        print('Loading data...')

        # save np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

        (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_features, index_from=3)

        # restore np.load for future normal usage
        np.load = np_load_old

        word_to_id = imdb.get_word_index()
        word_to_id ={k:(v+3) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        id_to_word = {value:key for key,value in word_to_id.items()}

        print(len(x_train), 'train sequences')
        print(len(x_val), 'test sequences')

        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
        y_train = np.eye(2)[y_train]
        y_val = np.eye(2)[y_val] 

        np.save('./data/x_train.npy', x_train)
        np.save('./data/y_train.npy', y_train)
        np.save('./data/x_val.npy', x_val)
        np.save('./data/y_val.npy', y_val)
        with open('data/id_to_word.pkl','wb') as f:
            pickle.dump(id_to_word, f)    

    else:
        x_train, y_train, x_val, y_val = np.load('data/x_train.npy'),np.load('data/y_train.npy'),np.load('data/x_val.npy'),np.load('data/y_val.npy')
        with open('data/id_to_word.pkl','rb') as f:
            id_to_word = pickle.load(f)

    return x_train, y_train, x_val, y_val, id_to_word

###########################################
###############Original Model##############
###########################################

def create_original_model():
    """
    Build the original model to be explained. 

    """
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def get_original_model():
    """
    Build the original model to be explained. 

    """
    inputs = Input((maxlen,))
    emb=Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen)(inputs)
    x=Dropout(0.2)(emb)
    x=Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1)(x)
    x=GlobalMaxPooling1D()(x)
    x=Dense(hidden_dims)(x)
    x=Dropout(0.2)(x)
    x=Activation('relu')(x)
    logit=Dense(2)(x)
    pred=Activation('softmax')(logit)
    
    model =Model(inputs, pred)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    y = Input((2,))
    
    model_pred = Model(inputs, [pred, emb])

    return model,model_pred

def get_sep_model():
    """
    Build the original model to be explained. 

    """
    inputs = Input((maxlen,))
    emb=Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen)(inputs)
    emb_model =Model(inputs, emb)

    emb_model.compile(loss='mse',
                  optimizer='adam')
    
    inputs_emb = Input((maxlen,embedding_dims))
    
    
    x=Dropout(0.2)(emb)
    x=Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1)(x)
    x=GlobalMaxPooling1D()(x)
    x=Dense(hidden_dims)(x)
    x=Dropout(0.2)(x)
    x=Activation('relu')(x)
    logit=Dense(2)(x)
    pred=Activation('softmax')(logit)
    
    cnn_model =Model(inputs_emb, pred)

    cnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
 

    return emb_model,cnn_model

def gradient_explain(M, x, y):
    out, emb = M(x)    
    loss = tf.reduce_sum(out*y)
    grad = tf.gradients(loss, emb)
    abs_grad = tf.abs(grad[0]*emb)
    grad_sum = tf.reduce_sum(abs_grad, axis=-1)
    return grad_sum
 
def calculate_acc(pred, y):
    return np.mean(np.argmax(pred, axis = 1) == np.argmax(y, axis = 1))

def generate_original_preds(train = True): 
    """
    Generate the predictions of the original model on training
    and validation datasets. 

    The original model is also trained if train = True. 

    """
    x_train, y_train, x_val, y_val, id_to_word = load_data() 
    model = create_original_model()

    if train:
        filepath="models/original.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
            verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)

    model.load_weights('./models/original.hdf5', 
        by_name=True) 

    pred_train = model.predict(x_train,verbose = 1, batch_size = 1000)
    pred_val = model.predict(x_val,verbose = 1, batch_size = 1000)
    if not train:
        print('The val accuracy is {}'.format(calculate_acc(pred_val,y_val)))
        print('The train accuracy is {}'.format(calculate_acc(pred_train,y_train)))


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

# Merge_prior = Lambda(lambda x: (x[2]*x[0]+x[1])/(x[2]+1.), 
#     output_shape=lambda x: x[0]) 
Merge_prior = Lambda(lambda x: x[0], 
    output_shape=lambda x: x[0]) 
 
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
        logits_ = K.permute_dimensions(logits, (0,2,1))# [batch_size, 1, d]
        
        d = int(logits_.get_shape()[2])
        unif_shape = [batch_size,self.k,d]
        
        uniform=K.random_uniform(shape=unif_shape,minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)
 
        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis = 1) 
        
        logits = tf.reshape(logits,[-1, d]) 
 
        
        output = K.in_train_phase(samples, logits) 
        return tf.expand_dims(output,-1)

    def compute_output_shape(self, input_shape):
        return input_shape

def construct_gumbel_selector(X_ph, num_words, embedding_dims, maxlen,y=None):
    """
    Build the MEED model for selecting words. 

    """
    emb_layer = Embedding(num_words, embedding_dims, input_length = maxlen, name = 'emb_gumbel')
    emb = emb_layer(X_ph)  
    net = Dropout(0.2, name = 'dropout_gumbel')(emb)
    net = emb
    first_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(net)    

    # global info
    net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(first_layer)
    global_info = Dense(100, name = 'new_dense_1', activation='relu')(net_new) 
    
    if y is not None:
        hy = Dense(100)(y)
        hy = Dense(100, activation='relu')(hy)
        hy = Dense(100, activation='relu')(hy)

    # local info
    net = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(first_layer) 
    local_info = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(net)  
    if y is not None:
        global_info = concatenate([global_info,hy])
        combined = Concatenate()([global_info,local_info]) 
    else:
        combined = Concatenate()([global_info,local_info]) 
    net = Dropout(0.2, name = 'new_dropout_2')(combined)
    net = Conv1D(100, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)   

    logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net)  
    
    return logits_T

def get_explainer(maxlen,max_features, embedding_dims,k):
    X_ph = Input(shape=(maxlen,), dtype='int32')
    y = Input(shape=(2,))
    logits_T_prior = Input(shape=(maxlen,1))
    step = Input(shape=(1,1))

    logits_T_self = construct_gumbel_selector(X_ph, max_features, embedding_dims, maxlen,y=y)
    logits_T = Merge_prior([logits_T_self,logits_T_prior,step])
    logits_T_output = concatenate([logits_T_self,logits_T],axis=1)
    
    tau = 0.5 
    T = Sample_Concrete(tau, k)(logits_T)
    T_neg = Sample_Concrete(tau, k)(Negative(logits_T))
    
    E = Model([X_ph,y,logits_T_prior,step], T)
    opt = Adam()  
    E.compile(loss='mse', optimizer=opt)   
    
    E_neg = Model([X_ph,y,logits_T_prior,step], T_neg)
    opt = Adam()  
    E_neg.compile(loss='mse', optimizer=opt)  
    
    E_soft = Model([X_ph,y,logits_T_prior,step], logits_T_output)
    opt = Adam()  
    E_soft.compile(loss='mse', optimizer=opt)  
    
    E_pred = Model([X_ph,y], logits_T_self)
    opt = Adam()  
    E_pred.compile(loss='mse', optimizer=opt)  
     
    return E, T, E_neg,T_neg,E_soft,E_pred

def get_approximator_eval(maxlen,max_features, embedding_dims):
    
    X_ph = Input(shape=(maxlen,), dtype='int32')
 
    
    emb2 = Embedding(max_features, embedding_dims, 
            input_length=maxlen)(X_ph)
     
    x = Dropout(0.2)(emb2)
    x = Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(hidden_dims)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    preds = Activation('softmax')(x)
 
    A = Model(X_ph, preds)
    opt = RMSprop()  
    A.compile(loss='categorical_crossentropy', optimizer=opt)   
    
    return A 

def get_approximator(maxlen,max_features, embedding_dims):
    
    X_ph = Input(shape=(maxlen,), dtype='int32')
    T = Input(shape=(maxlen,1))
    
    emb2 = Embedding(max_features, embedding_dims, 
            input_length=maxlen)(X_ph)
    
    x = Multiply()([emb2, T])
    
    x = Dropout(0.2)(x)
    x = Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(hidden_dims)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    preds = Activation('softmax')(x)
 
    
    A = Model([X_ph,T], preds)
    opt = Adam()  
    A.compile(loss='mse', optimizer=opt)   
    
    
    return A,preds
 

def get_model_pos(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,E_soft,logits_T_prior,step):

    set_trainability(E, False) 
    set_trainability(As, True) 
    set_trainability(Au, True) 

    T = E([X_ph,y,logits_T_prior,step])
    T_neg = Invert(T) 
    preds_s = As([X_ph,T])
    preds_u = Au([X_ph,T_neg])
     
    output_set = [preds_s,preds_u]
     
    model = Model(inputs=[X_ph,y,logits_T_prior,step],outputs=output_set)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    
    return model,output_set

def get_model_neg(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,E_soft,logits_T_prior,step):

    set_trainability(E, True) 
    set_trainability(As, False) 
    set_trainability(Au, False) 

    T = E([X_ph,y,logits_T_prior,step])
    T_neg = Invert(T) 
    preds_s = As([X_ph,T])
    preds_u = Au([X_ph,T_neg])
    
    logit_T_output = E_soft([X_ph,y,logits_T_prior,step])
    
    output_set = [preds_s,preds_u,logit_T_output]
     
    model = Model(inputs=[X_ph,y,logits_T_prior,step],outputs=output_set)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    
    return model,output_set

def neg_categorical_crossentropy(y_true, y_pred):
    return -K.mean(K.categorical_crossentropy(y_true, y_pred))

def loss_prior(y_true, y_pred):
    m = y_true
    d = my_get_shape(y_pred)/2
    log_z_self,log_z_final = y_pred[:,:d,:],y_pred[:,d:,:]
    z_final = K.softmax(log_z_final,axis=1)
    z_final = K.stop_gradient(z_final)
    loss = -K.sum(z_final*log_z_self,axis=1)/(m+1.)
    return K.mean(loss)
 
def build_model(maxlen,max_features, embedding_dims,k):
  
    X_ph = Input(shape=(maxlen,), dtype='int32')
    y = Input(shape=(2,))
    logits_T_prior = Input(shape=(maxlen,1))
    step = Input(shape=(1,1))
    E,_,E_neg,_,E_soft,E_pred=get_explainer(maxlen,max_features, embedding_dims,k)   
    As,_=get_approximator(maxlen,max_features, embedding_dims)
    Au,_=get_approximator(maxlen,max_features, embedding_dims)
  
    opt = RMSprop()
  
    loss_weights = [1.,1.]
    loss = ['categorical_crossentropy','categorical_crossentropy']
    model_pos,_ = get_model_pos(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,E_soft,logits_T_prior,step)
 
    loss_weights = [1.,1.,0]
    loss = ['categorical_crossentropy','categorical_crossentropy',loss_prior]
    model_neg,_ = get_model_neg(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,E_soft,logits_T_prior,step)
 
    return model_pos,model_neg,E,As,Au,E_pred


def MEED(train = True): 
 
    
    epochs = 20
    batch_size = 32
    flag_train_app = True

    print('Loading dataset...') 
    x_train, y_train, x_val, y_val, id_to_word = load_data()
    
    
    with session as sess:
        original_model, model_predict = get_original_model()
        if False:
            original_model.fit(x_train, y_train, validation_data=(x_val, y_val),
                               epochs=epochs, batch_size=batch_size)
            original_model.save('models/given_model_train.h5')
            model_predict.save('models/given_model_pred.h5')
        else:
            
            original_model = load_model('models/given_model_train.h5')
            model_predict = load_model('models/given_model_pred.h5')
                       
        M = original_model
 
    
        pred_train = M.predict(x_train,batch_size=256)
        pred_val = M.predict(x_val,batch_size=256)
       

        x_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, maxlen])
        y_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        grad_sum = gradient_explain(model_predict, x_placeholder, y_placeholder)
        
         
        print('Creating model...')
        model_pos,model_neg,_,As,Au,E = build_model(maxlen,max_features, embedding_dims,k)

        train_acc = np.mean(np.argmax(pred_train, axis = 1)==np.argmax(y_train, axis = 1))
        val_acc = np.mean(np.argmax(pred_val, axis = 1)==np.argmax(y_val, axis = 1))
        print('The train and validation accuracy of the original model is {} and {}'.format(train_acc, val_acc))

        if train:
             
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

                    y_batch = M.predict(x_batch)
                    y_batch_one_hot = to_categorical(np.argmax(y_batch,axis=1),2)
                    
                    selection_prior = sess.run(grad_sum, feed_dict={x_placeholder: x_batch, y_placeholder: y_batch_one_hot})
 
                    selection_prior = softmax(-selection_prior)
                    selection_prior = np.log(selection_prior+1e-40)
                    selection_prior = selection_prior[:,:,np.newaxis]
                    epoch_cur = np.ceil(i_step*batch_size/float(x_train.shape[0]))
                    epoch_cur = epoch_cur * np.ones((x_batch.shape[0],1,1))
 
                 
                    model_pos.train_on_batch([x_batch,y_batch,selection_prior,epoch_cur],[y_batch,y_batch])
                    model_neg.train_on_batch([x_batch,y_batch,selection_prior,epoch_cur],[y_batch,1.-y_batch,epoch_cur])

                 
                if i_step in step_list:
 
                    print('------------------------ test ----------------------------')
                    pred_train = M.predict(x_train,batch_size = 2048)
                    
                    st = time.time()
                    pred_val = M.predict(x_val,batch_size = 2048)
                    duration = time.time() - st
                    print('TPS = {}'.format(duration/x_val.shape[0]))    
               
                    fidelity,infidelity = eval_without_approximator(E, M, 
                                                                    x_val,pred_val,flag_with_y=True,k=k,
                                                                    id_to_word=id_to_word)
                    print('step: %d\tFS-M=%.4f\tFU-M=%.4f'%(i_step,fidelity,infidelity))
  
                    if flag_train_app:
 
                        model_As = get_approximator_eval(maxlen,max_features, embedding_dims)
                        model_Au = get_approximator_eval(maxlen,max_features, embedding_dims)

                        fidelity,infidelity,model_As,model_Au = eval_with_approximator(E, model_As,model_Au, 
                                                                     x_train, pred_train, x_val, pred_val, epochs=5,
                                                                     flag_with_y=True,k=k)
                        print('step: %d\tFS-A=%.4f\tFU-A=%.4f'%(i_step,fidelity,infidelity))
                         
 
    return  

if __name__ == '__main__':
 

    import os,signal,traceback
    try:
        MEED()
    except:
        traceback.print_exc()
    finally:
        os.kill(os.getpid(),signal.SIGKILL)

    





