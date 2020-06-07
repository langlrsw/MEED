# -*- coding: utf-8 -*-

try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
import keras

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

import tensorflow as tf
 
    
def eval_fidelity(model_As, selection, x_test, y_test,flag_only_mask=False,flag_show_example=False,id_to_word=None,idx_pos=None,idx_neg=None,flag_user_idx=False):
    if flag_only_mask:
        selected_x = selection
    else:
        selected_x = selection * x_test
    y_pred = model_As.predict(selected_x,batch_size = 2048)
    
    if flag_user_idx == False:
    
        idx_pos = np.where(np.argmax(y_test, axis=-1) == np.argmax(y_pred, axis=-1))[0]
        idx_neg = np.where(np.argmax(y_test, axis=-1) != np.argmax(y_pred, axis=-1))[0]
 
#     print('idx_neg',idx_neg)
    
    if flag_show_example:
        if flag_user_idx == False:
            np.random.shuffle(idx_pos)
#         idx_pos = np.array([ 3580, 15370, 23023, 13813,   389,   502,  9764,  1248, 17205,
#        19509])
        print('idx_pos[:10]',idx_pos[:10])
    
        if flag_user_idx == False:
            np.random.shuffle(idx_neg)

#         idx_neg = np.array([ 5629, 17540,  8711, 23822, 17890,  6884, 13056,  7703,   910,
#        10162])
        print('idx_neg[:10]',idx_neg[:10])
        
        for i in range(10):
            j = idx_pos[i]
            print('fidelity, consistent, order=%d, y_test = %d, y_pred = %d' % (j,np.argmax(y_test[j]),np.argmax(y_pred[j])))
            word = [id_to_word.get(x_test[j][e],'-1') for e in np.where(selection[j]==1)[0]]
            print(word)
            sent = ''
            for e in x_test[j]:
                if e > 1:
                    word = id_to_word.get(e,'-1')
                    if word!='-1':
                
                        sent = sent + ' ' + word.encode("utf-8")
            print(sent)
        
        for _ in range(10):
            print('****************')

        for i in range(10):
            j = idx_neg[i]
            print('fidelity, inconsistent, order=%d, y_test = %d, y_pred = %d' % (j,np.argmax(y_test[j]),np.argmax(y_pred[j])))
            word = [id_to_word.get(x_test[j][e],'-1') for e in np.where(selection[j]==1)[0]]
            print(word)
            sent = ''
            for e in x_test[j]:
                if e > 1:
                    word = id_to_word.get(e,'-1')
                    if word!='-1':
                
                        sent = sent + ' ' + word.encode("utf-8")
            print(sent)

    test_acc = np.mean(np.argmax(y_test, axis=-1) == np.argmax(y_pred, axis=-1))

    return test_acc


def eval_infidelity(model_Au, selection, x_test, y_test,flag_only_mask=False,flag_show_example=False,id_to_word=None):
    selection = np.ones_like(selection) - selection
    
    if flag_only_mask:
        selected_x = selection
    else:
        selected_x = selection * x_test
    y_pred = model_Au.predict(selected_x,batch_size = 2048)
    
    if flag_show_example:
        idx = np.arange(x_test.shape[0])
        np.random.shuffle(idx)
        for i in range(20):
            print('infidelity, y_test = %d, y_pred = %d' % (np.argmax(y_test[idx[i]]),np.argmax(y_pred[idx[i]])))
            word = [id_to_word.get(e,'-1') for e in np.where(selection[idx[i]]==0)[0]]
            print(word)

    test_acc = np.mean(np.argmax(y_test, axis=-1) == np.argmax(y_pred, axis=-1))
    return test_acc


from scipy.special import logsumexp

def softmax(x,axis=1):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def soft2hard_explanation(scores,k,selection_prior=None):
    
    if selection_prior is not None:
        scores = np.log(softmax(scores)+1e-40)
        selection_prior = np.log(softmax(selection_prior)+1e-40)
        scores = scores + selection_prior
        
     
    new_scores = np.zeros(scores.shape)
    scores = np.reshape(scores,[scores.shape[0],np.prod(scores.shape[1:])])
    selected = np.argsort(scores,axis=1)[:,-k:] 
    for i in range(scores.shape[0]):
        new_scores[i][selected[i]] = 1.0
    
    return new_scores

def eval_correlation(selection_1, selection_2, flag_abs=False):
    # Spearman Correlation Coefficient
    selection_1, selection_2 = np.reshape(selection_1, -1), np.reshape(selection_2, -1)
    if flag_abs:
        selection_1, selection_2 = np.abs(selection_1), np.abs(selection_2)

    temp_1, temp_2 = selection_1.argsort(), selection_2.argsort()
    ranks_1, ranks_2 = np.empty_like(temp_1), np.empty_like(temp_2)
    ranks_1[temp_1], ranks_2[temp_2] = np.arange(len(selection_1)), np.arange(len(selection_2))

    return np.corrcoef(selection_1, selection_2)[0][1]
 
def eval_without_approximator(model_explainer, model_predictor, x_test, y_test,flag_with_y=False,k=10,flag_only_mask=False,flag_show_example=False,id_to_word=None,selection=None,selection_prior=None,idx_pos=None,idx_neg=None,flag_user_idx=False):
 
    
    if selection is None:
        if flag_with_y:
            selection = model_explainer.predict([x_test,y_test],batch_size = 2048)
        else:
            selection = model_explainer.predict(x_test,batch_size = 2048)
            
    if selection_prior is not None:
        selection = np.reshape(selection,x_test.shape)
        selection_prior = np.reshape(selection_prior,x_test.shape)
        selection = soft2hard_explanation(selection,k,selection_prior=selection_prior)
    
    else:
        if len(x_test.shape) == 2:
            selection = np.reshape(selection,x_test.shape)
        selection = soft2hard_explanation(selection,k)
    acc_fidelity = eval_fidelity(model_predictor, selection, x_test, y_test,flag_only_mask=flag_only_mask,flag_show_example=flag_show_example,id_to_word=id_to_word,idx_pos=idx_pos,idx_neg=idx_neg,flag_user_idx=flag_user_idx)
    acc_infidelity = eval_infidelity(model_predictor, selection, x_test, y_test,flag_only_mask=flag_only_mask)

    return acc_fidelity, acc_infidelity



def eval_with_approximator(model_explainer, model_As,model_Au, x_train, y_train, x_test, y_test, epochs=5,flag_train=True,flag_with_y=False,k=10,flag_only_mask=False,flag_eval_train=False,flag_train_test=False,selection_train=None,selection_test=None):
 

    if selection_train is None:
        if flag_with_y:
            selection_train = model_explainer.predict([x_train,y_train],batch_size=2048) 
            selection_test = model_explainer.predict([x_test,y_test],batch_size=2048) 
        else:
            selection_train = model_explainer.predict(x_train,batch_size=2048) 
            selection_test = model_explainer.predict(x_test,batch_size=2048) 
        
    if len(x_train.shape) == 2:
        selection_train = np.reshape(selection_train,x_train.shape)
        selection_test = np.reshape(selection_test,x_test.shape)
    
    selection_train = soft2hard_explanation(selection_train,k)
    selection_test = soft2hard_explanation(selection_test,k)
 
    if flag_train:
        
#         print('fidelity: np.min(np.sum(x_train_f==0,axis=1))',np.min(np.sum(x_train_f==0,axis=1)))
        if flag_train_test:
            model_As.fit(x_test*selection_test, y_test, epochs=epochs,verbose=0)
        else:
            model_As.fit(x_train*selection_train, y_train, epochs=epochs,verbose=0)

    if flag_eval_train:
        acc_fidelity = eval_fidelity(model_As, selection_train, x_train, y_train,flag_only_mask=flag_only_mask)
    else:
        acc_fidelity = eval_fidelity(model_As, selection_test, x_test, y_test,flag_only_mask=flag_only_mask)
 
    if flag_train:
        
        
#         print('infidelity: np.min(np.sum(x_train_if==0,axis=1))',np.min(np.sum(x_train_if==0,axis=1)))
        if flag_train_test:
            model_Au.fit(x_test*(1.-selection_test), y_test, epochs=epochs,verbose=0)
        else:
            model_Au.fit(x_train*(1 - selection_train), y_train, epochs=epochs,verbose=0)

    if flag_eval_train:
        acc_infidelity = eval_infidelity(model_Au, selection_train, x_train, y_train,flag_only_mask=flag_only_mask)
    else:
        acc_infidelity = eval_infidelity(model_Au, selection_test, x_test, y_test,flag_only_mask=flag_only_mask)

    return acc_fidelity, acc_infidelity,model_As,model_Au


def eval(model_explainer, ori_predictor, get_approximator, x_train, y_train, x_test, y_test, epochs=10):

    f_original, if_original = eval_without_approximator(model_explainer, ori_predictor, x_test, y_test)

    f_approximator, if_approximator = eval_with_approximator(model_explainer, get_approximator, x_train, y_train, x_test, y_test, epochs)
    return f_original, if_original, f_approximator, if_approximator


def get_model():
    input = Input([191])
    out = Dense(2, activation='softmax')(input)
    return Model(input, out)

if __name__ == '__main__':
    import keras
    from keras.models import Model
    from keras.layers import Dense, Input
    from keras.utils import to_categorical

    n = 10000
    selection = np.random.randn(n, 191)
    selection_ones = np.ones_like(selection)

    x_train = np.random.randn(n, 191)
    x_test = np.random.randn(n, 191)

    y = to_categorical(np.random.randint(0, 2, size=(n,)), 2)
    y_selector = to_categorical(np.random.randint(0, 2, size=(n,)), 191)

    input = Input([191])
    out = Dense(2, activation='softmax')(input)

    predictor = Model(input, out)
    out_2 = Dense(2, activation='softmax')(input)
    predictor_2 = Model(input, out_2)


    predictor.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    predictor_2.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    predictor.fit(x_train, y, epochs=1)



    out_2 = Dense(191, activation='softmax')(input)
    selector = Model(input, out_2)
    selector.compile(loss=keras.losses.binary_crossentropy,
                     optimizer=keras.optimizers.Adadelta(),
                     metrics=['accuracy'])
    selector.fit(x_train, y_selector, epochs=1)



    y_pred_train = predictor.predict(x_train)
    y_pred_test = predictor.predict(x_test)

    # selection = selector.predict(x)
    # print(eval_fidelity(predictor, selection, x_test, y_pred_test))
    # print(eval_fidelity(predictor_2, selection, x_test, y_pred_test))

    f_original, if_original, f_approximator, if_approximator = eval(selector, predictor, get_model, x_train, y_pred_train, x_test, y_pred_test, 1)

    print(f_original)
    print(if_original)
    print(f_approximator)
    print(if_approximator)
